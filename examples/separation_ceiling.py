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
        choices=("rim", "flows", "both"),
        default="both",
        help="which signal's ceiling to measure; both is one run over one set of pictures",
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

    wants_rim = arguments.separation in ("rim", "both")
    wants_flows = arguments.separation in ("flows", "both")

    separated, flooded, naive = [], [], []
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
    print(
        f"  connected_components         {components['mean_average_precision']:.4f}   "
        "<- the bar any model using it is capped by"
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
