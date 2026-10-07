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
import json
import urllib.request
import zipfile
from pathlib import Path

from pyplatypus import Engine, from_dict, summarise_cases

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


def specification(train: str, validation: str, epochs: int, size: int) -> dict:
    """The experiment, as the spec the engine takes.

    Two things worth noticing. The masks arrive one file per nucleus, which the data layer unites
    - that is the layout this dataset is known for. And `include_background=False` on the metric,
    because most of a microscopy image is background and averaging it in turns a model that found
    nothing into one that scores 0.9.
    """
    return {
        "task": "semantic_segmentation",
        "data": {
            "train_path": train,
            "validation_path": validation,
            "colormap": [[0, 0, 0], [255, 255, 255]],
            "mode": "config_file",
        },
        "models": [
            {
                "name": "dsbowl-unet",
                "architecture": "u_net",
                "input_shape": [size, size],
                "channels": 3,
                "blocks": 4,
                "filters": 16,
                "batch_size": 8,
                "epochs": epochs,
                "loss": {"name": "cce_dice"},
                "metrics": [
                    {"name": "dice", "include_background": False},
                    {"name": "iou", "include_background": False},
                ],
                "optimizer": {"name": "adam", "learning_rate": 0.001},
                "callbacks": [
                    {
                        "name": "reduce_lr_on_plateau",
                        "monitor": "val_loss",
                        "factor": 0.5,
                        "patience": 4,
                    },
                    {
                        "name": "early_stopping",
                        "monitor": "val_loss",
                        "patience": 10,
                        "restore_best": True,
                    },
                ],
                "augmentation": [
                    # A transform's own arguments live under `params`, which is what keeps the spec
                    # free of albumentations' API: the names are checked against the installed
                    # version, the parameters are passed through untouched.
                    {"name": "HorizontalFlip", "params": {"p": 0.5}},
                    {"name": "VerticalFlip", "params": {"p": 0.5}},
                    {"name": "RandomRotate90", "params": {"p": 0.5}},
                    {"name": "RandomBrightnessContrast", "params": {"p": 0.3}},
                ],
            }
        ],
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
        trained_with=f"pyplatypus {__import__('pyplatypus').__version__}",
        split={"fractions": [0.8, 0.2], "seed": arguments.seed, **split["samples"]},
        validation=distribution,
        worst_cases=[{"case": row["case"], "dice": row["dice"]} for row in worst],
    )
    print(f"\nwrote {path}")
    print(f"      {path.with_suffix('.json')}")
    print("\nUpload both files, then add the commit to REGISTRY in pyplatypus/weights.py.")


if __name__ == "__main__":
    main()
