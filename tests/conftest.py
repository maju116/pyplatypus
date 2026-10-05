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
    # `task` is required of every configuration, so the shared fixture states it rather
    # than relying on a default that no longer exists.
    return {"task": "semantic_segmentation", "data": data_block, "models": [model_block]}


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


# --- detection -------------------------------------------------------------------------

#: The two classes the synthetic detection fixtures use. Not blood cells: a fixture that
#: looks like real data invites reading real conclusions off it.
DETECTION_CLASSES = ["square", "bar"]

#: Deliberately not square and not the model's size, so every test below exercises the
#: letterbox rather than an identity transform. 160x128 into a 128x128 model pads the top
#: and bottom, which is the case that hides a transposed axis.
SOURCE_SHAPE = (128, 160)


def write_voc_sample(root, key, boxes, labels, *, shape=SOURCE_SHAPE,
                     declared_shape=None, subdirs=("images", "annotations")):
    """One sample in nested_dirs layout: a PNG and a Pascal VOC XML beside it.

    Coordinates are written in VOC's own convention - 1-based and inclusive of both ends -
    because that is what the files a user has actually contain, and a fixture written
    0-based would let a reader off the one conversion that matters.

    `declared_shape` lets a test write an annotation that disagrees with its image, which
    is a real failure: a dataset gets resized and the XML is copied along unchanged.
    """
    height, width = shape
    image = np.full((height, width, 3), 30, np.uint8)
    for (x0, y0, x1, y1), label in zip(boxes, labels):
        image[int(y0):int(y1), int(x0):int(x1)] = 220 if label == 0 else 120

    sample = Path(root) / key
    (sample / subdirs[0]).mkdir(parents=True, exist_ok=True)
    (sample / subdirs[1]).mkdir(parents=True, exist_ok=True)
    write_png(sample / subdirs[0] / f"{key}.png", image)

    said_h, said_w = declared_shape if declared_shape is not None else (height, width)
    objects = "".join(
        f"<object><name>{DETECTION_CLASSES[label]}</name><bndbox>"
        f"<xmin>{int(x0) + 1}</xmin><ymin>{int(y0) + 1}</ymin>"
        f"<xmax>{int(x1)}</xmax><ymax>{int(y1)}</ymax>"
        f"</bndbox></object>"
        for (x0, y0, x1, y1), label in zip(boxes, labels)
    )
    (sample / subdirs[1] / f"{key}.xml").write_text(
        f"<annotation><size><width>{said_w}</width><height>{said_h}</height>"
        f"<depth>3</depth></size>{objects}</annotation>"
    )
    return sample


def detection_split(root, prefix, count, seed):
    """Enough variety in the box shapes that fitting anchors has something to fit.

    k-means refuses when there are fewer distinct shapes than anchors, correctly, and a
    fixture of identical boxes would hit that rather than the thing under test.
    """
    rng = np.random.default_rng(seed)
    for n in range(count):
        x0, y0 = (int(v) for v in rng.integers(4, 50, 2))
        side = int(rng.integers(16, 40))
        bar = int(rng.integers(28, 58))
        write_voc_sample(
            root, f"{prefix}_{n}",
            [(x0, y0, x0 + side, y0 + side), (96, 20, 96 + bar, 34)],
            [0, 1],
        )
    return root


@pytest.fixture
def detection_root(tmp_path):
    """A train and a validation split of synthetic VOC-annotated images.

    Under a subdirectory of its own so that a test can use this and `data_block` at once:
    both would otherwise make `tmp_path/train` and the second one to run would fail on a
    directory that already exists.
    """
    root = tmp_path / "detection"
    detection_split(root / "train", "train", 8, seed=0)
    detection_split(root / "valid", "valid", 4, seed=1)
    return root


@pytest.fixture
def detection_config(detection_root):
    """The smallest detection specification that trains: two epochs, two anchors."""
    return {
        "task": "object_detection",
        "seed": 1,
        "data": {
            "train_path": str(detection_root / "train"),
            "validation_path": str(detection_root / "valid"),
            "classes": DETECTION_CLASSES,
        },
        "models": [{
            "name": "d",
            "input_shape": [128, 128],
            "epochs": 2,
            "batch_size": 2,
            "anchors_per_grid": 2,
            "optimizer": {"name": "adam", "learning_rate": 1e-3},
        }],
    }


@pytest.fixture
def voc_sample():
    """The writer above, as a fixture.

    A test module *can* import from here - `tests/data/test_dsbowl.py` does
    `from tests.conftest import DSBOWL` - so this is a preference and not a necessity. The
    first version of this docstring said it was impossible, which `test_dsbowl.py`
    disproves two directories away. A fixture is used anyway because pytest resolves it by
    name without a path, and because a relative import across `tests/` genuinely does not
    work: that is the form that failed, and the conclusion drawn from it was too broad.
    """
    return write_voc_sample


@pytest.fixture
def detection_classes():
    return list(DETECTION_CLASSES)


@pytest.fixture
def source_shape():
    """(height, width) of the synthetic images: not square, and not the model's size."""
    return SOURCE_SHAPE


@pytest.fixture(scope="session")
def make_detection_split():
    """`detection_split`, for a test that needs a split of its own.

    Session-scoped because it hands back a function and holds no state, and because a
    module-scoped fixture cannot ask for a function-scoped one - which is what training a
    detector once per file needs.
    """
    return detection_split
