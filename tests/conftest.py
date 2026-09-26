from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyplatypus.spec.data import SegmentationData

DSBOWL = Path("examples/data/data_science_bowl")


def write_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


@pytest.fixture
def data_block(tmp_path):
    train = tmp_path / "train"
    valid = tmp_path / "valid"
    train.mkdir()
    valid.mkdir()
    return {
        "train_path": str(train),
        "validation_path": str(valid),
        "colormap": [[0, 0, 0], [255, 255, 255]],
        "mode": "nested_dirs",
    }


@pytest.fixture
def model_block():
    return {"name": "unet", "input_shape": [256, 256], "n_class": 2}


@pytest.fixture
def config(data_block, model_block):
    return {"data": data_block, "models": [model_block]}


@pytest.fixture
def nested_root(tmp_path):
    """Three samples laid out the way the Data Science Bowl does it: one image, several
    binary mask files per sample."""
    root = tmp_path / "train"
    for n in range(3):
        sample = root / f"sample_{n}"
        write_png(sample / "images" / f"{n}.png", np.full((64, 64, 3), 10 * n, np.uint8))
        for m in range(2):
            mask = np.zeros((64, 64, 3), np.uint8)
            mask[m * 20:(m + 1) * 20, :] = 255
            write_png(sample / "masks" / f"{m}.png", mask)
    return root


@pytest.fixture
def binary_data():
    return SegmentationData(
        train_path="unused", validation_path="unused",
        colormap=[(0, 0, 0), (255, 255, 255)],
    )


@pytest.fixture
def volume_root(tmp_path):
    """Three samples of NIfTI volumes with label-map masks, laid out as nested_dirs.

    Small on purpose - 8x8x4 - because a 3D test that takes a minute gets skipped, and a
    skipped test proves nothing.
    """
    nib = pytest.importorskip("nibabel")
    root = tmp_path / "volumes"
    for n in range(3):
        sample = root / f"case_{n}"
        (sample / "images").mkdir(parents=True, exist_ok=True)
        (sample / "masks").mkdir(parents=True, exist_ok=True)

        scan = np.full((8, 8, 4), -1000.0, dtype=np.float32)      # air
        scan[2:6, 2:6, 1:3] = 40.0 + 10 * n                       # soft tissue
        labels = np.zeros((8, 8, 4), dtype=np.float32)
        labels[2:6, 2:6, 1:3] = 1

        affine = np.diag([1.0, 1.0, 2.5, 1.0])
        nib.save(nib.Nifti1Image(scan, affine), str(sample / "images" / "scan.nii.gz"))
        nib.save(nib.Nifti1Image(labels, affine), str(sample / "masks" / "labels.nii.gz"))
    return root


@pytest.fixture
def volume_data():
    return SegmentationData(
        train_path="unused", validation_path="unused", labels=[0, 1],
    )
