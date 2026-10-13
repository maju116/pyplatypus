"""Does a flow field beat a rim once it is trained, measured against the bar on one set of
pictures.

    python examples/train_instance_flows.py

The second half of `#212`. The first half (`0.8.0a18`) measured the ceilings on 134 held-out
Data Science Bowl cases: **flows at 0.9924** against the rim route's **0.9807**, both far above
the **0.7583** where connected components caps any model. A ceiling is not a result, and
`#159`'s whole lesson is that a route with a fine ceiling can still turn over - the rim's
ceiling was 0.9807 and the trained model reached **0.5587**, which is the number to beat.

**This script did not go through `Engine`, and that was a decision rather than a shortcut -
which `#160` has since settled.** A flow model's head is `2 + n_class` wide and the engine
took its head width from `spec.data.n_class`, so naming the axis before the number existed
would have promised a model nobody had measured, which `§8` forbids in as many words. The
number exists now and the axis is named: `task: instance_segmentation` derives the width
from `separation`, whose block carries `mode`, `loss` and `weight` - the three this script
takes as command-line arguments - and `examples/compare_instance_separations.py` runs both
signals from one configuration file.

So this script is **the record of the measurement above and no longer the way to run one**.
It is kept because a negative result nobody can reproduce is an opinion: every row of that
table came out of this file, assembled from public pieces - `build_model`, `Mse`, `CceDice`,
`AlbumentationsAugmenter`, `flows_from_instances`, `instances_by_flows` - and rewriting it to
go through the engine would mean the table was produced by code that no longer exists.

**The data path is `#210`'s and not a way around it.** The *instance* label map is what the
augmenter sees, as integers; the flow target is built from what comes back. A target built
before augmentation comes back geometrically consistent and physically wrong, and nothing
downstream can tell. That is why the dataset below reads per-object files with
`read_instances` instead of `unite_masks`, which ORs instance identity away at read time.

**The split is the directory split already on disk**, 536 training cases and the 134 held out,
and that is said rather than assumed because of what the first half found: a ceiling of 0.9846
and a bar of 0.7655 are on the record for "the 134 held-out cases" and no set of cases here
reproduces them. These numbers are on the same 134 the ceilings above were measured on, and
the case and object counts are printed with them.

Four columns, and three of them exist so a moved instance score can be attributed rather than
celebrated - the lesson of `§4bo`, where a table only became readable once the mask's own Dice
sat beside it:

    foreground Dice     did the mask move, or did the separation?
    flow angle          how wrong the predicted vectors are where the truth has one
    AP flows            the number this script is for
    AP cc               the same predicted mask through connected components, which is what
                        the signal has to beat to have done anything at all

And then the pair that attributes what is left, `#211`'s trick: substitute the truth for
exactly **one** part of the pipeline and read the difference, which costs one more decoding
and no training. Perfect vectors with this mask, and this field with a perfect mask. The
first of them is `truth_field_on` and not `flows_from_instances(labels)`, and the difference
between those two is the whole of this script's second lesson - read that function.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from pyplatypus.data.augmentation import AlbumentationsAugmenter
from pyplatypus.instances import (
    flows_from_instances,
    instances_by_flows,
    instances_from_mask,
    read_instances,
    score_cases,
    summarise_instances,
)
from pyplatypus.models import build_model
from pyplatypus.objectives.losses import CceDice, Mse
from pyplatypus.spec import SegmentationModel
from pyplatypus.spec.components import AugmentationStep

#: What the card for `dsbowl-unet` records. The architecture, the size, the optimiser and the
#: epoch count are held to it so the only things that differ are the target and the head.
PUBLISHED = {"blocks": 4, "filters": 16, "size": 256, "batch_size": 32, "learning_rate": 1e-3}

#: Flips and right angles only. Every one of them is exact under `#210`'s rule anyway, but the
#: rule is that the target is *rebuilt* rather than corrected, so nothing here depends on that.
TRANSFORMS = [
    AugmentationStep(name="HorizontalFlip", params={"p": 0.5}),
    AugmentationStep(name="VerticalFlip", params={"p": 0.5}),
    AugmentationStep(name="RandomRotate90", params={"p": 0.5}),
]


def model_spec(size: int) -> SegmentationModel:
    """The model half of a specification, built directly rather than through `from_dict`.

    `from_dict` validates the data paths, and rightly - it is just that nothing here asks the
    spec for data, so the only honest thing to hand it would be paths it would then never
    read. The head width is passed to `build_model` separately, which **was** the line the
    engine could not write - `#160` has since derived it, so
    `examples/compare_instance_separations.py` is the path to copy and this script is the
    record of the measurement it produced. Kept rather than rewritten: the numbers in the
    docstring above came out of this file, and a negative result nobody can reproduce is an
    opinion.
    """
    return SegmentationModel(
        name="flows",
        architecture="u_net",
        channels=3,
        blocks=PUBLISHED["blocks"],
        filters=PUBLISHED["filters"],
        input_shape=[size, size],
        batch_size=PUBLISHED["batch_size"],
        epochs=1,
        loss={"name": "cce_dice"},
    )


def resize_labels(labels: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour to the training size, then relabelled consecutively.

    An object can vanish here, and one that vanished must not leave a hole in the numbering:
    the metric counts objects by their values, so a gap would read as a missing object that the
    truth itself does not contain.
    """
    small = np.asarray(
        Image.fromarray(labels.astype(np.int32), mode="I").resize(
            (size, size), Image.Resampling.NEAREST
        )
    )
    present = np.unique(small[small > 0])
    lookup = np.zeros(int(small.max()) + 1, dtype=np.int32)
    lookup[present] = np.arange(1, present.size + 1, dtype=np.int32)
    return lookup[small]


def read_case(case: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    """One case as `(image HWC float32 in 0..1, instance label map HW int32)`."""
    files = sorted((case / "images").iterdir())
    with Image.open(files[0]) as handle:
        picture = handle.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    image = np.asarray(picture, dtype=np.float32) / 255.0
    return image, resize_labels(read_instances(case), size)


class Cases(torch.utils.data.Dataset):
    """Images and instance label maps, augmented as integers, target built afterwards.

    The ordering is the whole point and it is the one thing a comment cannot guarantee, which
    is why `tests/data/test_vector_targets.py` pins it with a recording augmenter. Here it is
    visible in four lines: augment, then build.
    """

    def __init__(self, cases: list[Path], size: int, *, augment: bool, mode: str):
        self.cases = cases
        self.size = size
        self.mode = mode
        self.augmenter = AlbumentationsAugmenter(TRANSFORMS) if augment else None
        self._read: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index not in self._read:
            self._read[index] = read_case(self.cases[index], self.size)
        image, labels = self._read[index]

        if self.augmenter is not None:
            image, labels = self.augmenter(image, labels)

        target = build_target(labels, self.mode)
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            torch.from_numpy(target),
        )


def build_target(labels: np.ndarray, mode: str) -> np.ndarray:
    """`(4, H, W)`: two flow channels, then the two-class foreground one-hot.

    The order is the contract and the decode below depends on it. The foreground channels are
    an ordinary semantic target; the flow channels are undefined off the objects, which is why
    the loss on them is masked and why the mask comes from **channel 3** rather than from the
    flow's own zeros - a flow field is exactly zero at every sink, so its non-zero voxels are
    the foreground minus precisely the seeds.
    """
    flow = flows_from_instances(labels, mode=mode)
    foreground = (labels > 0).astype(np.float32)
    return np.concatenate([flow, (1.0 - foreground)[None], foreground[None]], axis=0)


def angles_between(predicted: np.ndarray, truth: np.ndarray, where: np.ndarray) -> np.ndarray:
    """Degrees between two fields, on the voxels the truth defines one for."""
    if not where.any():
        return np.zeros(0, dtype=np.float32)
    a = predicted[:, where]
    b = truth[:, where]
    a = a / np.maximum(np.hypot(a[0], a[1]), 1e-12)
    b = b / np.maximum(np.hypot(b[0], b[1]), 1e-12)
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=0), -1.0, 1.0)))


def truth_field_on(labels: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """The best field perfect object identity could give, over the support `mask` asks about.

    The first version of this oracle handed `flows_from_instances(labels)` straight to the
    predicted mask, and what came out was a reading of the field's **domain** rather than of
    the field: 0.2220 against 0.5315 for the model's own field, with **18758 objects back
    against 5999 in the truth**. A flow field is identically zero off its own objects and a
    zero field moves nothing, so every voxel the mask got wrong stood still and became a
    basin of its own. Confirmed without a model, because the mechanism does not need one -
    dilating the truth mask by a single pixel took 48 objects to 93, one spurious ring each,
    and a second pixel added nothing because the ring is still one component.

    So the identity is what gets substituted, which is what the oracle was always for. Every
    voxel of the mask joins the nearest truth object - `distance_transform_edt`'s indices,
    the same Voronoi step `instances_from_parts` uses - and the field is built from that.

    The domain is `mask | truth foreground` rather than `mask` alone, and the extra voxels
    cost nothing because `follow_flows` only ever *starts* from the mask. What they buy is
    that every truth object keeps its whole body, so the diffusion has a connected core to
    spread from; on `mask` alone an object the mask had cut in two would hand one of the
    pieces a zero field again, which is the artefact this function exists to remove.

    A fragment of mask that touches no truth object anywhere still joins the nearest one by
    identity, which is the honest answer rather than a generous one: a perfect field cannot
    know that a false positive is one, and the cost lands where it belongs, in the
    intersection-over-union of the object it was glued to. Whether it gets a *vector*,
    though, is the target's business and not this function's - a straight line to the sink
    needs no path, while the diffusion is heat spread inside the object and a piece
    disconnected from its body receives none. So under `diffusion` such a piece keeps a zero
    vector and lands as one basin of its own, and the run **counts them and prints the count
    beside the row** rather than leaving the reader to wonder which of the two it is looking
    at. That is the whole lesson of the number this function replaced: an object count that
    nothing accounts for is not a measurement.
    """
    foreground = labels > 0
    if not foreground.any():
        # No test catches the removal of these two lines and that is said rather than
        # papered over: without them an empty truth still comes back as a zero field of the
        # right shape, because `distance_transform_edt` of an all-background picture answers
        # index 0 everywhere and `labels[0, 0]` is 0. Measured, not assumed - and it is an
        # implementation detail of one function rather than anything scipy documents, which
        # is the whole reason the branch stays.
        return np.zeros((2, *labels.shape), dtype=np.float32)
    _, indices = ndimage.distance_transform_edt(~foreground, return_indices=True)
    nearest = labels[tuple(indices)]
    nearest[~(np.asarray(mask) | foreground)] = 0
    return flows_from_instances(nearest, mode=mode)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/data_science_bowl/stage1_train")
    parser.add_argument("--validation", default="examples/data/data_science_bowl/stage1_validation")
    parser.add_argument("--mode", choices=("centroid", "diffusion"), default="diffusion")
    parser.add_argument("--epochs", type=int, default=60, help="60 is what the card records")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--size", type=int, default=PUBLISHED["size"])
    parser.add_argument(
        "--flow-weight",
        type=float,
        default=1.0,
        dest="flow_weight",
        help="how much the masked regression term counts against the semantic one. CellPose "
        "uses 5 on a field scaled differently from this one, so 1.0 here is a starting "
        "point and not a replication",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--step-size", type=float, default=1.0, dest="step_size")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N cases")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse()
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)

    train_cases = sorted(p for p in Path(arguments.data).iterdir() if p.is_dir())
    held_out = sorted(p for p in Path(arguments.validation).iterdir() if p.is_dir())
    if arguments.limit:
        train_cases = train_cases[: arguments.limit]
        held_out = held_out[: arguments.limit]
    if not train_cases or not held_out:
        raise SystemExit("no case directories under one of the two paths")

    device = torch.device(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Two flow channels plus the two semantic ones. This was the line the engine could not
    # write; `InstanceModel.head_channels` derives it now (`#160`), which is why this script
    # is the record of a measurement rather than the way to run one.
    model = build_model(model_spec(arguments.size), n_class=4).to(device)

    train = torch.utils.data.DataLoader(
        Cases(train_cases, arguments.size, augment=True, mode=arguments.mode),
        batch_size=PUBLISHED["batch_size"],
        shuffle=True,
        num_workers=arguments.workers,
        drop_last=False,
    )

    regression = Mse(where="foreground")
    semantic = CceDice()
    optimizer = torch.optim.Adam(model.parameters(), lr=PUBLISHED["learning_rate"])

    print(f"{len(train_cases)} training cases, {len(held_out)} held out, device {device}")
    print(f"target: flows:{arguments.mode}, head 4 channels, flow weight {arguments.flow_weight}")
    started = time.perf_counter()
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        totals = np.zeros(3)
        batches = 0
        for images, target in train:
            images = images.to(device)
            target = target.to(device)
            mask = target[:, 3:4]
            logits = model(images)
            flow_loss = regression(logits[:, :2], target[:, :2], mask)
            mask_loss = semantic(logits[:, 2:], target[:, 2:])
            loss = arguments.flow_weight * flow_loss + mask_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += [loss.item(), flow_loss.item(), mask_loss.item()]
            batches += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == arguments.epochs:
            mean = totals / max(batches, 1)
            print(
                f"  epoch {epoch:3d}/{arguments.epochs}  loss {mean[0]:.4f}"
                f"  flow {mean[1]:.4f}  mask {mean[2]:.4f}"
                f"  {time.perf_counter() - started:6.0f}s"
            )

    print("\nscoring the held-out cases")
    model.eval()
    flows_rows, components_rows = [], []
    # Two oracle rows, which cost one extra decoding each and no training at all. #211's
    # trick: substitute the ground truth for exactly one part of the pipeline and read the
    # difference, rather than arguing about which part is the limit. Together they bracket
    # the attribution - a perfect field with this model's mask, and this model's field with
    # a perfect mask - and the pair says which half is worth working on.
    true_field_rows, true_mask_rows = [], []
    dice_total = 0.0
    angle_medians: list[float] = []
    objects = 0
    # The oracle's one remaining blind spot, counted rather than argued about. A piece of
    # predicted foreground containing no truth voxel at all joins the nearest object by
    # identity, but a diffusion field is heat spread *within* the object, so a piece
    # disconnected from that object's body never receives any and keeps a zero vector. Each
    # such piece lands as one basin, so this count is how much of the oracle's object count
    # is not the oracle's doing. Under `centroid` it is zero by construction - a straight
    # line to the sink needs no path - which is why the mode is printed beside it.
    detached_pieces = 0
    for index, case in enumerate(held_out, start=1):
        image, labels = read_case(case, arguments.size)
        objects += int(labels.max())
        with torch.no_grad():
            logits = model(
                torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None].to(device)
            )[0]
        flow = logits[:2].cpu().numpy()
        probability = torch.softmax(logits[2:], dim=0).cpu().numpy()
        predicted_foreground = probability[1] > 0.5

        truth_foreground = labels > 0
        pieces, found = ndimage.label(predicted_foreground, structure=np.ones((3, 3), bool))
        detached_pieces += found - len({int(v) for v in np.unique(pieces[truth_foreground])} - {0})
        overlap = float((predicted_foreground & truth_foreground).sum())
        dice_total += (
            2 * overlap / float(predicted_foreground.sum() + truth_foreground.sum())
            if predicted_foreground.sum() + truth_foreground.sum()
            else 1.0
        )
        truth_flow = flows_from_instances(labels, mode=arguments.mode)
        defined = truth_foreground & (np.hypot(truth_flow[0], truth_flow[1]) > 0)
        degrees = angles_between(flow, truth_flow, defined)
        if degrees.size:
            angle_medians.append(float(np.median(degrees)))

        flows_rows.append(
            (
                case.name,
                labels,
                instances_by_flows(
                    flow,
                    predicted_foreground,
                    steps=arguments.steps,
                    step_size=arguments.step_size,
                ),
            )
        )
        components_rows.append((case.name, labels, instances_from_mask(predicted_foreground)))
        true_field_rows.append(
            (
                case.name,
                labels,
                instances_by_flows(
                    truth_field_on(labels, predicted_foreground, arguments.mode),
                    predicted_foreground,
                    steps=arguments.steps,
                    step_size=arguments.step_size,
                ),
            )
        )
        true_mask_rows.append(
            (
                case.name,
                labels,
                instances_by_flows(
                    flow,
                    truth_foreground,
                    steps=arguments.steps,
                    step_size=arguments.step_size,
                ),
            )
        )
        if index % 25 == 0:
            print(f"  {index}/{len(held_out)}", flush=True)

    with_flows = summarise_instances(score_cases(flows_rows))
    with_components = summarise_instances(score_cases(components_rows))
    with_true_field = summarise_instances(score_cases(true_field_rows))
    with_true_mask = summarise_instances(score_cases(true_mask_rows))
    dice = dice_total / len(held_out)

    print(f"\n--- {len(held_out)} cases, {objects} objects: the set these numbers are on ---")
    print(f"  foreground Dice            {dice:.4f}   <- did the mask move?")
    print(
        f"  flow angle, median of case medians  {np.median(angle_medians):.1f} degrees"
        if angle_medians
        else "  flow angle                 no voxels"
    )

    # Every row carries the number of objects it handed back, beside the truth's count in
    # the header. That is not decoration: the first version of the TRUE-field oracle scored
    # 0.2220 and returned 18758 objects against 5999, and the count is what said at a glance
    # that the row was measuring the field's domain rather than the field. An average
    # precision alone cannot distinguish a route that mislabels from one that shatters.
    def row(label: str, summary: dict, note: str = "") -> None:
        print(
            f"  {label:<26}{summary['mean_average_precision']:.4f}"
            f"   {summary['predicted_objects']:>6} objects back{note}"
        )

    row("AP flows", with_flows)
    row("AP connected components", with_components, "   <- same mask, signal discarded")
    print("\n--- the two oracles, each one part of the pipeline replaced by the truth ---")
    row("TRUE identity, this mask", with_true_field, "   <- what perfect vectors would buy")
    row("this field, TRUE mask", with_true_mask, "   <- what a perfect mask would buy")
    print(
        "\n  Diagnostics, not routes. Whichever of the two is further above the AP flows row\n"
        "  is the half worth working on, and if neither is far above it the limit is the\n"
        "  decoding rather than either input - which #211 measured for the rim route, where\n"
        "  a perfect surface was worth 0.003.\n\n"
        "  The first row is TRUE IDENTITY and not the truth's own field, which is a\n"
        "  correction rather than a wording: the truth's field is zero wherever the mask is\n"
        "  wrong, a zero field moves nothing, and the row read 0.2220 with 18758 objects\n"
        "  back before `truth_field_on` gave every voxel of the mask the nearest truth\n"
        "  object to point at. An oracle has to replace one part of the pipeline and leave\n"
        "  the rest alone; that one was quietly replacing the mask's support as well."
    )
    print(f"\n  pieces of predicted foreground holding no truth voxel  {detached_pieces}")
    print(
        "  Each is one basin the TRUE-identity row returns that the field did not earn:\n"
        "  diffusion is heat spread inside an object, so a piece not connected to that\n"
        "  object's body receives none and keeps a zero vector. Read the row's object\n"
        "  count against this number."
        if arguments.mode == "diffusion"
        else "  Under centroid these cost nothing: a straight line to the sink needs no path\n"
        "  through the object, so every one of them gets a vector. The count is here to be\n"
        "  compared with the diffusion run's."
    )
    # The reference rows were measured on the 134 directories of stage1_validation and 6000
    # objects. Saying so is the whole lesson of the first half: a ceiling and a bar that
    # travel without their set of pictures are what produced the 0.9846 and 0.7655 nobody
    # can reproduce. So the header states the set, and says plainly when this run is not on
    # it rather than printing a comparison that is not one.
    # The references are at **this** resolution, measured on the resized truth, because that
    # is what the model sees. The native-resolution ceilings differ - 0.9924, 0.9807 and
    # 0.7583 against the three below - and putting a 256-pixel score beside a native one is
    # this milestone's own recurring mistake in a new costume.
    same_pictures = len(held_out) == 134 and objects == 5999
    print("\n--- what these have to be read against: 134 cases, 5999 objects, at 256x256 ---")
    if not same_pictures:
        print(
            f"  !! this run scored {len(held_out)} cases and {objects} objects, so the four\n"
            "     rows below are NOT on these pictures and the comparison is not one"
        )
    print("  the flows post-process ceiling, on the resized truth    0.9873")
    print("  the rim post-process ceiling, on the resized truth      0.9629")
    print("  connected components on the resized truth, the bar      0.7505")
    print("  the trained rim model, on ITS OWN cases, not these      0.5587")
    print(
        "\n  The last row is the number to beat and it is on no set measured here: #159's runs\n"
        "  used a seeded fifth of stage1_train and the first half of #212 could not\n"
        "  reproduce the set its ceiling was quoted on. Read it as the order of magnitude to\n"
        "  clear, and read the connected-components row above as the honest comparison -\n"
        "  same model, same mask, same pictures, with and without the flow signal."
    )

    if arguments.out:
        Path(arguments.out).write_text(
            json.dumps(
                {
                    "mode": arguments.mode,
                    "epochs": arguments.epochs,
                    "flow_weight": arguments.flow_weight,
                    "cases": len(held_out),
                    "objects": objects,
                    "on_the_reference_pictures": same_pictures,
                    "foreground_dice": dice,
                    "flow_angle_median": float(np.median(angle_medians)) if angle_medians else None,
                    "flows": with_flows,
                    "connected_components": with_components,
                    "oracle_true_field": with_true_field,
                    "oracle_true_mask": with_true_mask,
                    "held_out": [case.name for case in held_out],
                },
                indent=1,
            )
        )
        print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
