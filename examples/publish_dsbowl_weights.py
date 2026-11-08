"""Train the published `dsbowl-unet` weights, from the data's own source.

    python examples/publish_dsbowl_weights.py --out weights/

Downloads BBBC038v1 from the Broad Institute, trains a U-Net on it, measures the result per
image, and writes `dsbowl-unet.safetensors` with the sidecar that records all of it. The file is
then ready to upload; this script deliberately does not upload anything.

**Why not the Kaggle copy.** The images are the same, but the Kaggle download is governed by the
competition rules you accept to get it, while BBBC038v1 carries an explicit CC0 waiver: "the
various contributors of the imagesets have waived all copyright and related or neighboring rights
to BBBC038v1". Weights are a derivative of the data they were trained on, so taking the data from
the source with the clear licence is what makes them publishable without an argument.

Citation, which CC0 does not require and which is owed anyway:

    Caicedo et al., "Nucleus segmentation across imaging experiments: the 2018 Data Science
    Bowl", Nature Methods, 2019. Image set BBBC038v1, Broad Bioimage Benchmark Collection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from pyplatypus import Engine, from_dict, plot_masks, summarise_cases

SOURCE = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"
CITATION = (
    "Caicedo et al., Nucleus segmentation across imaging experiments: the 2018 Data Science "
    "Bowl, Nature Methods, 2019. Image set BBBC038v1, Broad Bioimage Benchmark Collection."
)


def fetch(into: Path) -> Path:
    """Download and unpack stage1_train, unless it is already there."""
    into.mkdir(parents=True, exist_ok=True)
    archive = into / "stage1_train.zip"
    unpacked = into / "stage1_train"

    if unpacked.is_dir() and any(unpacked.iterdir()):
        print(f"already unpacked: {unpacked}")
        return unpacked

    if not archive.exists():
        print(f"downloading {SOURCE} (83 MB)")
        urllib.request.urlretrieve(SOURCE, archive)

    print(f"unpacking into {unpacked}")
    unpacked.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(unpacked)
    return unpacked


def _comparison() -> object:
    """The architecture comparison, loaded by path, because it owns this recipe.

    `pyplatypus#446`. This script stated the `dsbowl-unet` model field for field and
    `compare_dsbowl_architectures.py` stated it again as the `u_net` row of its table - 61
    shared literals, and nothing held the two together. Measured: the learning rate changed
    in either copy alone left 318 tests passing.

    `#418` exempted this pair on a reason that was not true - *"no example trains that
    model, so it is the only copy rather than a second one"* - and `ARCHITECTURES` begins
    with `u_net`. An example does train it, and it is the first row of the table the
    published weights are judged by.

    The direction is `#418`'s own: `publish_fives_weights.py` loads
    `segment_retinal_vessels.py` and says why - *"drift from the run the example documents
    is the one thing this script must not do"*. The example owns the recipe; the publisher
    borrows it.
    """
    path = Path(__file__).with_name("compare_dsbowl_architectures.py")
    spec = importlib.util.spec_from_file_location("compare_dsbowl_architectures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: Everything this script says about the model that the comparison does not: the published
#: name, and the batch size the comparison takes as an argument. Two keys, and a reader can
#: see the whole difference between the two runs here rather than by diffing two files.
PUBLISHED_OVERRIDES = {"name": "dsbowl-unet", "batch_size": 8}


def specification(train: str, validation: str, epochs: int, size: int) -> dict:
    """The experiment, as the spec the engine takes.

    Two things worth noticing. The masks arrive one file per nucleus, which the data layer unites
    - that is the layout this dataset is known for. And `include_background=False` on the metric,
    because most of a microscopy image is background and averaging it in turns a model that found
    nothing into one that scores 0.9.

    The model itself comes from `compare_dsbowl_architectures.model("u_net", ...)` - see
    `_comparison()` for why that direction.
    """
    model = dict(
        _comparison().model(
            "u_net", epochs=epochs, size=size, batch=PUBLISHED_OVERRIDES["batch_size"]
        )
    )
    model.update(PUBLISHED_OVERRIDES)
    return {
        "task": "semantic_segmentation",
        "data": {
            "train_path": train,
            "validation_path": validation,
            "colormap": [[0, 0, 0], [255, 255, 255]],
            "mode": "config_file",
        },
        "models": [model],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="examples/data/bbbc038", help="where to download and unpack the images"
    )
    parser.add_argument("--out", default="weights", help="where to write the weights")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--figures", default=None, help="a directory to draw the worst validation cases into"
    )
    arguments = parser.parse_args()

    root = fetch(Path(arguments.data))

    # Split held out by image, with a seed, so the reported numbers are on data the model never
    # saw and the same split comes back on any machine. Each image here is its own case - the
    # dataset is unrelated fields of view, not slices of one patient - so no grouping is needed.
    from pyplatypus import split_dataset

    split = split_dataset(
        root, Path(arguments.data) / "splits", fractions=(0.8, 0.2), seed=arguments.seed
    )
    print("split:", json.dumps(split["samples"]))

    spec = from_dict(
        specification(
            split["paths"]["train"], split["paths"]["validation"], arguments.epochs, arguments.size
        )
    )
    engine = Engine(spec, device=arguments.device, num_workers=arguments.workers)
    engine.fit(verbose=True)

    print("\n--- the split as one number ---")
    for row in engine.evaluate():
        print(
            {
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in row.items()
            }
        )

    print("\n--- per image ---")
    cases = engine.evaluate_cases("dsbowl-unet")
    distribution = summarise_cases(cases)
    for row in distribution:
        print(
            f"  {row['metric']}: n={row['n']} mean={row['mean']:.4f} "
            f"sd={row['sd']:.4f} median={row['median']:.4f} min={row['min']:.4f} "
            f"worst={row['worst_case']}"
        )

    worst = sorted(cases, key=lambda row: row["dice"])[:5]
    for row in worst:
        print(f"  worst: {row['case'][:40]} dice={row['dice']:.3f}")

    if arguments.figures:
        # The worst four rather than the first four. A figure of the best cases is an
        # advertisement, and the card these weights ship with reports a worst case of 0.72,
        # so this is the picture that corresponds to the number a stranger will read.
        #
        # `evaluate_cases` walks the split in the dataset's own order and never shuffles, so
        # a row's position in the list is the index to draw.
        figures = Path(arguments.figures)
        figures.mkdir(parents=True, exist_ok=True)
        ranked = sorted(range(len(cases)), key=lambda index: cases[index]["dice"])[:4]
        dataset = engine.dataset(engine.runs["dsbowl-unet"].spec, "validation")
        figure = plot_masks(
            np.stack([dataset[index][0] for index in ranked]),
            prediction=engine.predict("dsbowl-unet", "validation")[ranked],
            truth=np.stack([dataset[index][1] for index in ranked]),
            labels=[f"dice {cases[index]['dice']:.3f}" for index in ranked],
        )
        figure.savefig(figures / "dsbowl-worst.png", dpi=110, bbox_inches="tight")
        print(f"\nwrote {figures / 'dsbowl-worst.png'}")

    # Everything a stranger needs to judge these weights travels with them. The metrics are the
    # validation distribution rather than the mean alone, because a mean hides the failures and
    # somebody deciding whether to use these is entitled to see them.
    path = engine.export_weights(
        "dsbowl-unet",
        Path(arguments.out) / "dsbowl-unet",
        data="BBBC038v1 (2018 Data Science Bowl), stage1_train",
        data_source=SOURCE,
        data_licence="CC0 1.0",
        citation=CITATION,
        weights_licence="MIT",
        split={"fractions": [0.8, 0.2], "seed": arguments.seed, **split["samples"]},
        validation=distribution,
        worst_cases=[{"case": row["case"], "dice": row["dice"]} for row in worst],
    )
    print(f"\nwrote {path}")
    print(f"      {path.with_suffix('.json')}")
    print("\nUpload both files, then add the commit to REGISTRY in pyplatypus/weights.py.")


if __name__ == "__main__":
    main()
