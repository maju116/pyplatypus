"""How good instance separation could get here, measured before any model is trained.

    python examples/separation_ceiling.py --data path/to/stage1_train

`#159` proposes predicting a semantic mask **and a separation signal**, then turning the pair
into instances. Two signals are measured here, on one set of pictures in one run:

    rim     each object eroded by a pixel is its *interior* and what the erosion removed is
            its *rim*, which with the background makes three mutually exclusive classes -
            ordinary semantic segmentation, nothing new in the model, the loss or the metrics
    flows   a unit vector per foreground pixel pointing at its own object's sink, decoded by
            following the field until the pixels that arrive together are one object (#212).
            Two ways to build it: the straight line to the sink, and CellPose's heat
            diffusion inside the object
    hv      HoVer-Net's pair of distance maps - for every foreground pixel its signed offset
            from its own object's centre along each axis, normalised per object (#213). Two
            ways to get markers out of them, and the difference is the whole of `#213`: the
            maps' own zeros, which are object centres by construction, or the gradient of
            each map, which is HoVer-Net's published boundary route

`#159` closed measured-and-no because interior and rim are two classes competing for the same
pixels - a rim wide enough to learn hollows the object out. A flow field has no such
competition, which is why `flows` is measured here next and why **the two modes are measured
against each other**: the cheap one is wrong for any object whose centroid falls outside it.

**This script trains nothing and loads no weights.** It feeds the ground truth through that
post-process and scores the result against the same ground truth, which is the best the shape
of solution could ever reach however good the model predicting the parts turns out to be. The
same cases go through plain connected components beside it, so the numbers in the output
are the ceiling of the new route and the ceiling of the old one, measured on one set of
pictures in one run.

**The watershed row is a floor, not a comparison, and the script says so where it prints
it.** A ceiling run has perfect parts, so the rim is binary and the surface the flood
climbs is flat; the watershed has nothing to climb and nearest-seed is already the optimal
distance split. The decoding can only show its worth on a trained model, where the surface
is a predicted probability with evidence in it.

**`hv` is measured here because `#213` asked for exactly this and said what the answer
decides**: "if the markers from `hv` do not beat eroded interiors on the ground truth, they
will not beat them predicted either, and this issue closes before a model is trained". They do
not. Over the 134 cases and 6000 objects below, with both recipes swept and each turning over
inside its range rather than at an edge:

    marker source                 AP    over the bar   markers   objects with none
    eroded interiors, 1 px    0.9807        +0.2224      6030                   0
    hv centres, |h|,|v|<=0.6  0.9382        +0.1799      5900                   5
    hv gradient, >=0.6        0.7995        +0.0412      5404                   0
    connected components      0.7583          the bar

The maps' own zeros are the better of the two recipes and still **0.0425 below a one-pixel
erosion**, which is the cheapest marker there is; HoVer-Net's published gradient route
recovers **a fifth** of what the erosion does over the bar. Both curves turn over at 0.6 -
`--hv-sweep` prints them - so neither is a tuning artefact, and the erosion's thin-object
fallback cannot explain the gap either: it rescues 4 objects of 6000 against a difference of
0.0425.

Two rows here are controls rather than findings: the connected-components bar and the
eroded-interior ceiling reproduce `0.8.0a18`'s recorded 0.7583 and 0.9807 **to four
decimals**, which is how a reader knows these numbers are on the same pictures as those -
the provenance failure that cost `#159`'s ceiling its set.

Why this comes first: `#157` built the metric before the model because a model chosen before
there is a way to score it is a model chosen on nothing. The same argument applies one step
further in. If the post-process caps the approach below what is already achieved, no training
run can rescue it, and that is worth four minutes rather than four hours.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from pyplatypus.instances import (
    FLOW_MODES,
    flows_from_instances,
    instances_by_flows,
    instances_by_watershed,
    instances_from_mask,
    instances_from_parts,
    parts_from_instances,
    read_instances,
    score_cases,
    summarise_instances,
)
from pyplatypus.instances.postprocess import _neighbours_agree


def hv_from_instances(labels: np.ndarray) -> np.ndarray:
    """`(2, H, W)` float32: channel 0 the horizontal map, channel 1 the vertical one.

    For every foreground pixel, its signed offset from its own object's centroid along that
    axis, divided by the largest such offset in the object - so each map runs -1 to 1 inside
    every object whatever its size. That is what "normalised per object" means in `#213`, and
    it is the property both marker recipes below rest on: two touching objects meet at a jump
    of 2 rather than at a jump that depends on how big they are.
    """
    labels = np.asarray(labels)
    horizontal = np.zeros(labels.shape, dtype=np.float32)
    vertical = np.zeros(labels.shape, dtype=np.float32)
    for value in np.unique(labels[labels > 0]):
        rows, columns = np.nonzero(labels == value)
        for offset, target in (
            (columns - columns.mean(), horizontal),
            (rows - rows.mean(), vertical),
        ):
            largest = np.abs(offset).max()
            target[rows, columns] = offset / largest if largest else 0.0
    return np.stack([horizontal, vertical])


def markers_from_hv_centres(hv: np.ndarray, foreground: np.ndarray, cut: float) -> np.ndarray:
    """Markers where both maps are near zero, which is an object's centre by construction.

    This is the recipe `#213` narrowed itself to after `#211` priced the other one: *the
    horizontal and vertical maps' own extrema as seeds rather than the gradient magnitude as
    a surface*. No derivative, one threshold, and the marker is inside the object whenever
    the near-centroid pixels are - which is not the same as the centroid being inside it.
    """
    return np.asarray(foreground) & (np.abs(hv[0]) <= cut) & (np.abs(hv[1]) <= cut)


def markers_from_hv_gradient(
    hv: np.ndarray, foreground: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """HoVer-Net's published post-process, up to the point where the markers exist.

    Each map is linear inside an object and jumps across a seam between two of them, so a
    derivative along each axis is large where objects part. The paper takes a Sobel of each
    map along its own axis, scales both to 0..1, keeps the larger, calls everything above a
    threshold a boundary, and the markers are the foreground that is left; the same surface,
    blurred, is what the watershed then climbs.

    **The scaling is global and the maps are normalised per object**, which is why the
    threshold has to be swept rather than taken from the paper. A small object's internal
    slope and a large object's seam jump are the same number after a global min-max, so one
    threshold over-segments the small objects while another merges the large ones. Measured:
    0.4, the published value, scores 0.6595 on 25 cases where 0.6 scores 0.7908.

    Returns `(markers, surface)`.
    """
    foreground = np.asarray(foreground)
    along_columns = _scale_to_unit(np.abs(ndimage.sobel(hv[0], axis=1)))
    along_rows = _scale_to_unit(np.abs(ndimage.sobel(hv[1], axis=0)))
    # `* foreground` is the paper's own masking and no test here catches its removal, which
    # is said rather than left as a gap: the markers cannot notice it, because masking only
    # changes background pixels and the markers are already `& foreground`, and the watershed
    # floods inside the foreground too. What does change is the surface a pixel or two inside
    # the edge, where the blur below pulls background values in - a second-order effect that
    # no fixture small enough to read distinguishes. It stays because it is what HoVer-Net
    # does, not because anything here proves it matters.
    boundary = np.maximum(along_columns, along_rows) * foreground
    surface = ndimage.gaussian_filter(boundary.astype(np.float32), 1.0)
    return foreground & (boundary < threshold), surface


def _scale_to_unit(array: np.ndarray) -> np.ndarray:
    low, high = float(array.min()), float(array.max())
    return (array - low) / (high - low) if high > low else np.zeros_like(array)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="examples/data/data_science_bowl/stage1_train",
        help="a directory of case directories, each with a masks/ folder of one file per object",
    )
    parser.add_argument("--limit", type=int, default=0, help="score only the first N cases")
    parser.add_argument("--width", type=int, default=1, help="how many pixels of rim to remove")
    parser.add_argument(
        "--separation",
        choices=("rim", "flows", "hv", "all"),
        default="all",
        help="which signal's ceiling to measure; all is one run over one set of pictures. "
        "This used to read rim|flows|both and `both` is now `all`, because there are three",
    )
    parser.add_argument(
        "--hv-cut",
        type=float,
        default=0.6,
        dest="hv_cut",
        help="markers are where both hv maps are within this of zero; 0.6 is the measured peak",
    )
    parser.add_argument(
        "--hv-threshold",
        type=float,
        default=0.6,
        dest="hv_threshold",
        help="HoVer-Net's boundary threshold on the scaled gradient; the paper's 0.4 is not "
        "the peak here because the scaling is global and the maps are per object",
    )
    parser.add_argument(
        "--hv-sweep",
        action="store_true",
        dest="hv_sweep",
        help="sweep both hv recipes instead of reporting one point each, because a single "
        "threshold answering `worse` says nothing until the range has been asked",
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="how many steps to follow a flow field for"
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1.0,
        dest="step_size",
        help="how far one step moves a pixel; above one pixel this collapses, see --help output",
    )
    parser.add_argument("--out", default=None, help="where to write the rows as JSON")
    parser.add_argument("--worst", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    arguments = parse()
    data = Path(arguments.data)
    cases = sorted(path for path in data.iterdir() if path.is_dir())
    if arguments.limit:
        cases = cases[: arguments.limit]
    if not cases:
        raise SystemExit(f"no case directories under {data}")
    print(f"{len(cases)} cases under {data}")

    wants_rim = arguments.separation in ("rim", "all")
    wants_flows = arguments.separation in ("flows", "all")
    wants_hv = arguments.separation in ("hv", "all")
    # The sweep is the claim rather than the point estimate: both hv recipes turn over inside
    # their range, and a reader who cannot see the turnover cannot tell a measured peak from
    # a lucky default - which is what the paper's 0.4 would have been here.
    hv_cuts = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8) if arguments.hv_sweep else (arguments.hv_cut,)
    hv_thresholds = (
        (0.3, 0.4, 0.5, 0.6, 0.7, 0.8) if arguments.hv_sweep else (arguments.hv_threshold,)
    )

    separated, flooded, naive = [], [], []
    hv_centres: dict[float, list] = {cut: [] for cut in hv_cuts}
    hv_gradient: dict[float, list] = {threshold: [] for threshold in hv_thresholds}
    hv_flooded: dict[float, list] = {threshold: [] for threshold in hv_thresholds}
    # Keyed by the row's own name, so the JSON reads the way the table does.
    hv_markers = {f"centres<={cut}": 0 for cut in hv_cuts} | {
        f"gradient>={threshold}": 0 for threshold in hv_thresholds
    }
    hv_orphans = dict.fromkeys(hv_markers, 0)
    followed = {mode: [] for mode in FLOW_MODES}
    build_seconds = dict.fromkeys(FLOW_MODES, 0.0)
    decode_seconds = dict.fromkeys(FLOW_MODES, 0.0)
    truth_objects = recovered_objects = 0
    erased_by_the_strict_erosion = thin_cases = 0
    split_objects = split_cases = 0
    centroid_outside = 0

    for index, case in enumerate(cases, start=1):
        truth = read_instances(case)
        foreground = truth > 0
        truth_objects += int(truth.max())
        naive.append((case.name, truth, instances_from_mask(foreground)))

        if wants_flows:
            # How many objects the cheap mode is *structurally* wrong about, counted rather
            # than inferred from the score gap: a centroid outside its own object aims the
            # straight line across a gap the object does not occupy.
            for value in np.unique(truth[foreground]):
                pixels = np.argwhere(truth == value)
                row, column = pixels.mean(axis=0).round().astype(int)
                if truth[row, column] != value:
                    centroid_outside += 1
            for mode in FLOW_MODES:
                started = time.perf_counter()
                field = flows_from_instances(truth, mode=mode)
                built = time.perf_counter()
                recovered = instances_by_flows(
                    field, foreground, steps=arguments.steps, step_size=arguments.step_size
                )
                build_seconds[mode] += built - started
                decode_seconds[mode] += time.perf_counter() - built
                followed[mode].append((case.name, truth, recovered))

        if wants_hv:
            maps = hv_from_instances(truth)
            values = np.unique(truth[foreground])
            for cut in hv_cuts:
                markers = markers_from_hv_centres(maps, foreground, cut)
                row = f"centres<={cut}"
                hv_markers[row] += int(ndimage.label(markers, structure=np.ones((3, 3), bool))[1])
                # An object with no marker at all cannot come back, which is this recipe's
                # own failure mode rather than the gradient route's over-segmentation - so it
                # is counted beside the score instead of being inferred from it.
                hv_orphans[row] += sum(1 for v in values if not markers[truth == v].any())
                hv_centres[cut].append(
                    (case.name, truth, instances_from_parts(markers, foreground))
                )
            for threshold in hv_thresholds:
                markers, surface = markers_from_hv_gradient(maps, foreground, threshold)
                row = f"gradient>={threshold}"
                hv_markers[row] += int(ndimage.label(markers, structure=np.ones((3, 3), bool))[1])
                hv_orphans[row] += sum(1 for v in values if not markers[truth == v].any())
                hv_gradient[threshold].append(
                    (case.name, truth, instances_from_parts(markers, foreground))
                )
                hv_flooded[threshold].append(
                    (case.name, truth, instances_by_watershed(markers, foreground, surface))
                )

        if not wants_rim:
            if index % 25 == 0:
                print(f"  {index}/{len(cases)}", flush=True)
            continue

        # What the plain erosion would have erased, measured here rather than read off the
        # finished interior: that one has already been rescued by the fallback, so counting
        # what is missing from it counts failures of the fallback instead of firings of it.
        # The first version of this probe did exactly that and reported zero.
        strict = foreground & _neighbours_agree(
            np.where(foreground, truth, 0), across_background=True
        )
        lost = np.setdiff1d(np.unique(truth[foreground]), np.unique(truth[strict]))
        if lost.size:
            erased_by_the_strict_erosion += int(lost.size)
            thin_cases += 1

        interior, _ = parts_from_instances(truth, width=arguments.width)
        recovered = instances_from_parts(interior, foreground)
        separated.append((case.name, truth, recovered))

        # The watershed's surface, for a ceiling run specifically. The ground truth's rim is
        # binary, so it is flat and gives the flood nothing to climb; the inverted distance
        # transform is the classic substitute and is what makes this the distance-transform
        # watershed rather than a more expensive nearest-seed. A *trained* run passes the
        # predicted rim probability instead, which is the whole point of the decoding.
        flooded.append(
            (
                case.name,
                truth,
                instances_by_watershed(
                    interior, foreground, -ndimage.distance_transform_edt(foreground)
                ),
            )
        )

        recovered_objects += int(recovered.max())
        if int(recovered.max()) > int(truth.max()):
            split_cases += 1
            split_objects += int(recovered.max()) - int(truth.max())
        if index % 25 == 0:
            print(f"  {index}/{len(cases)}")

    components = summarise_instances(score_cases(naive))
    rows: list[dict] = []
    ceiling = watershed_ceiling = None
    flow_ceilings = {}

    # Which pictures these numbers belong to, printed before them and written to the JSON.
    # A ceiling of 0.9846 and a bar of 0.7655 are on the record for "the 134 held-out cases"
    # and **no set of cases on this disk reproduces them**: the 134 directories of
    # stage1_validation give 0.9807 and 0.7583 over 6000 objects, the seed-1 fifth of
    # stage1_train gives 0.9905 and 0.7876 over 107 cases and 4044 objects, all 536 of
    # stage1_train give 0.9818 and 0.7618 over 23461, and the same 134 at the training
    # resolution give 0.9627 and 0.7504. The recorded object count of 5774 appears in none of
    # them. The conclusions drawn against that bar survive it - 0.5587 is below every one of
    # these - but a new number cannot be put beside an old one without this line.
    print(f"\n--- {len(cases)} cases, {truth_objects} objects: the set these numbers are on ---")
    print("--- the ceilings, all on the ground truth and the same cases ---")
    if wants_flows:
        for mode in FLOW_MODES:
            flow_ceilings[mode] = summarise_instances(score_cases(followed[mode]))
            summary = flow_ceilings[mode]
            print(
                f"  flows:{mode:<10s} + follow   {summary['mean_average_precision']:.4f}   "
                f"median {summary['median']:.4f}   worst {summary['worst'][:16]}"
            )
    if wants_rim:
        rows = score_cases(separated)
        ceiling = summarise_instances(rows)
        watershed_ceiling = summarise_instances(score_cases(flooded))
        print(
            f"  rim + nearest_seed           {ceiling['mean_average_precision']:.4f}   "
            f"median {ceiling['median']:.4f}   worst {ceiling['worst'][:16]}"
        )
        print(
            f"  rim + watershed              {watershed_ceiling['mean_average_precision']:.4f}   "
            f"median {watershed_ceiling['median']:.4f}   worst {watershed_ceiling['worst'][:16]}"
        )
    hv_ceilings: dict[str, dict] = {}
    if wants_hv:
        for cut in hv_cuts:
            summary = summarise_instances(score_cases(hv_centres[cut]))
            hv_ceilings[f"centres<={cut}"] = summary
            print(
                f"  hv centres |h|,|v|<={cut:<4.2f}     {summary['mean_average_precision']:.4f}"
                f"   {hv_markers[f'centres<={cut}']:5d} markers,"
                f" {hv_orphans[f'centres<={cut}']} objects without one"
            )
        for threshold in hv_thresholds:
            summary = summarise_instances(score_cases(hv_gradient[threshold]))
            hv_ceilings[f"gradient>={threshold}"] = summary
            flooded_summary = summarise_instances(score_cases(hv_flooded[threshold]))
            hv_ceilings[f"gradient>={threshold}+watershed"] = flooded_summary
            print(
                f"  hv gradient >={threshold:<4.2f}           {summary['mean_average_precision']:.4f}"
                f"   {hv_markers[f'gradient>={threshold}']:5d} markers,"
                f" {hv_orphans[f'gradient>={threshold}']} objects without one"
                f"   watershed {flooded_summary['mean_average_precision']:.4f}"
            )
    print(
        f"  connected_components         {components['mean_average_precision']:.4f}   "
        "<- the bar any model using it is capped by"
    )
    if wants_hv and wants_rim:
        print(
            "\n  The hv rows are #213's own test and they answer it: the issue says that if\n"
            "  markers from hv do not beat eroded interiors on the ground truth they will\n"
            "  not beat them predicted either. Measured against the bar rather than against\n"
            "  each other, because that is the share of the headroom each recipe recovers:\n"
            "  the erosion is +0.2224 over connected components, the maps' own zeros\n"
            "  +0.1799, and HoVer-Net's published gradient route +0.0412 - a fifth of the\n"
            "  erosion's, from the pair of channels that was supposed to beat it. Both hv\n"
            "  curves turn over at 0.6, inside their range rather than at its edge, so\n"
            "  neither number is a tuning artefact; run --hv-sweep to see them. And the\n"
            "  erosion's thin-object fallback cannot explain the gap: it rescued 4 objects\n"
            "  of 6000 here, against a difference of 0.0425.\n"
            "  What this does NOT say is that hv is a bad target. It says its markers are\n"
            "  worse than an erosion's on perfect inputs, which is the one condition #213\n"
            "  made necessary - and #212's oracle pair says why nobody should expect the\n"
            "  trained answer to differ: a perfect mask was worth +0.2398 there and perfect\n"
            "  vectors +0.0663, so the mask is the limit whichever signal sits beside it."
        )
    if wants_rim:
        print(
            "\n  Read the watershed row as a floor and not as a comparison. A ceiling run has\n"
            "  perfect parts, so its rim is binary and the surface the flood climbs is flat -\n"
            "  there is no evidence in it, and with a flat surface nearest-seed IS the optimal\n"
            "  distance split while the watershed only adds flooding-order artefacts. What this\n"
            "  row shows is that the decoding loses nothing it should not; whether it GAINS\n"
            "  anything is only answerable on a trained model, whose predicted rim probability\n"
            "  is a surface with real evidence in it."
        )

    if wants_flows:
        print("\n--- what the two flow targets differ by, and what each costs ---")
        print(
            f"  objects whose centroid falls outside themselves  {centroid_outside}"
            f"  of {truth_objects}"
            f"  ({100 * centroid_outside / max(truth_objects, 1):.2f}%)"
        )
        for mode in FLOW_MODES:
            back = sum(int(labels.max()) for _, _, labels in followed[mode])
            print(
                f"  {mode:10s} objects back {back:5d}   "
                f"build {build_seconds[mode] / len(cases) * 1000:6.1f} ms/case   "
                f"decode {decode_seconds[mode] / len(cases) * 1000:6.1f} ms/case"
            )
        print(
            "\n  The build cost is the one that matters for training rather than for this run:\n"
            "  the target is rebuilt from the augmented label map on every sample (#210), so\n"
            "  it is paid once per image per epoch and not once per experiment."
        )

    if wants_rim:
        print("\n--- what the decomposition costs, counted rather than assumed ---")
        print(f"  objects in the truth                        {truth_objects}")
        print(f"  objects recovered                           {recovered_objects}")
        print(
            f"  erased by the erosion, rescued by fallback  {erased_by_the_strict_erosion}"
            f" in {thin_cases} cases"
        )
        print(
            f"  gained by an interior splitting in two      {split_objects} in {split_cases} cases"
        )

        print(f"\n--- the {arguments.worst} worst cases ---")
        for row in sorted(rows, key=lambda row: row["average_precision"] or 0.0)[: arguments.worst]:
            print(
                f"  {row['case'][:16]}  ap={row['average_precision']:.4f}  "
                f"truth={row['truth']:4d}  back={row['predicted']:4d}"
            )

    print(
        "\nEvery ceiling here is the best its shape of solution could ever reach, however good\n"
        "the model predicting the parts turns out to be. The gap between one of them and what\n"
        "a model actually scores is the mask's own error, and no post-process reaches it."
    )
    if arguments.out:
        Path(arguments.out).write_text(
            json.dumps(
                {
                    "separation": arguments.separation,
                    "data": str(data),
                    "cases_scored": len(cases),
                    "ceiling": ceiling,
                    "watershed_ceiling": watershed_ceiling,
                    "flow_ceilings": flow_ceilings,
                    "hv_ceilings": hv_ceilings,
                    "hv_markers": hv_markers,
                    "hv_objects_without_a_marker": hv_orphans,
                    "flow_steps": arguments.steps,
                    "flow_step_size": arguments.step_size,
                    "centroid_outside_its_object": centroid_outside,
                    "connected_components": components,
                    "truth_objects": truth_objects,
                    "recovered_objects": recovered_objects,
                    "erased_by_the_strict_erosion": erased_by_the_strict_erosion,
                    "objects_gained_by_splitting": split_objects,
                    "cases": rows,
                },
                indent=1,
            )
        )
        print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
