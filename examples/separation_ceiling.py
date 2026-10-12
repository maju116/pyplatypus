"""How good instance separation could get here, measured before any model is trained.

    python examples/separation_ceiling.py --data path/to/stage1_train

`#159` proposes predicting a semantic mask **and a separation signal**, then turning the pair
into instances. The signal chosen is the simplest of the two on the table: each object eroded
by a pixel is its *interior* and what the erosion removed is its *rim*, which with the
background makes three mutually exclusive classes - ordinary semantic segmentation, nothing
new in the model, the loss or the metrics.

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
from pathlib import Path

import numpy as np
from scipy import ndimage

from pyplatypus.instances import (
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

    separated, flooded, naive = [], [], []
    truth_objects = recovered_objects = 0
    erased_by_the_strict_erosion = thin_cases = 0
    split_objects = split_cases = 0

    for index, case in enumerate(cases, start=1):
        truth = read_instances(case)
        foreground = truth > 0

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
        naive.append((case.name, truth, instances_from_mask(foreground)))

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

        truth_objects += int(truth.max())
        recovered_objects += int(recovered.max())
        if int(recovered.max()) > int(truth.max()):
            split_cases += 1
            split_objects += int(recovered.max()) - int(truth.max())
        if index % 25 == 0:
            print(f"  {index}/{len(cases)}")

    rows = score_cases(separated)
    ceiling = summarise_instances(rows)
    watershed_ceiling = summarise_instances(score_cases(flooded))
    components = summarise_instances(score_cases(naive))

    print("\n--- the ceilings, all on the ground truth and the same cases ---")
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
    print(
        "\n  Read the watershed row as a floor and not as a comparison. A ceiling run has\n"
        "  perfect parts, so its rim is binary and the surface the flood climbs is flat -\n"
        "  there is no evidence in it, and with a flat surface nearest-seed IS the optimal\n"
        "  distance split while the watershed only adds flooding-order artefacts. What this\n"
        "  row shows is that the decoding loses nothing it should not; whether it GAINS\n"
        "  anything is only answerable on a trained model, whose predicted rim probability\n"
        "  is a surface with real evidence in it."
    )

    print("\n--- what the decomposition costs, counted rather than assumed ---")
    print(f"  objects in the truth                        {truth_objects}")
    print(f"  objects recovered                           {recovered_objects}")
    print(
        f"  erased by the erosion, rescued by fallback  {erased_by_the_strict_erosion}"
        f" in {thin_cases} cases"
    )
    print(f"  gained by an interior splitting in two      {split_objects} in {split_cases} cases")

    print(f"\n--- the {arguments.worst} worst cases ---")
    for row in sorted(rows, key=lambda row: row["average_precision"] or 0.0)[: arguments.worst]:
        print(
            f"  {row['case'][:16]}  ap={row['average_precision']:.4f}  "
            f"truth={row['truth']:4d}  back={row['predicted']:4d}"
        )

    print(
        "\nThe gap between the two ceilings is what separation can buy. The gap between the\n"
        "lower one and what a model actually scores is the mask's own error, and no\n"
        "post-process reaches it."
    )
    if arguments.out:
        Path(arguments.out).write_text(
            json.dumps(
                {
                    "ceiling": ceiling,
                    "watershed_ceiling": watershed_ceiling,
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
