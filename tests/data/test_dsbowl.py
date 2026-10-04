"""The real thing: 536 training samples, 9 different image sizes, up to 27 mask files
per sample, all RGBA. Synthetic fixtures prove the logic; this proves the assumptions.

Skipped when the data is not on the machine, so CI stays green without a 5.6 GB download.
"""

import time

import numpy as np
import pytest

from pyplatypus.data import SegmentationDataset, discover, stitch, tile
from pyplatypus.spec.data import SegmentationData
from pyplatypus.spec.models import SegmentationModel
from tests.conftest import DSBOWL

pytestmark = pytest.mark.skipif(not DSBOWL.is_dir(), reason="Data Science Bowl data absent")

BINARY = [(0, 0, 0), (255, 255, 255)]


@pytest.fixture(scope="module")
def data():
    return SegmentationData(
        train_path=str(DSBOWL / "stage1_train"),
        validation_path=str(DSBOWL / "stage1_validation"),
        colormap=BINARY,
    )


@pytest.fixture(scope="module")
def train(data):
    return discover(data.train_path, data)


def test_every_training_sample_is_complete(train):
    """No warnings swallowed, no samples quietly missing."""
    assert len(train) == 536
    assert train.skipped == ()


def test_validation_split_is_there(data):
    assert len(discover(data.validation_path, data)) == 134


def test_samples_carry_many_mask_files(train):
    counts = [len(s.masks) for s in train.samples]
    assert max(counts) > 20          # one file per nucleus
    assert min(counts) >= 1


def test_the_binary_colormap_actually_describes_this_dataset(train, data):
    """If this drifts from zero the masks are not what the colormap claims, which is the
    quietest possible way to train on nothing."""
    model = SegmentationModel(name="unet", input_shape=(256, 256), n_class=2, blocks=4)
    dataset = SegmentationDataset(train.samples, model, data)
    assert dataset.colormap_coverage(limit=40) == pytest.approx(0.0, abs=1e-6)


def test_a_real_example_has_the_right_shape_and_content(train, data):
    model = SegmentationModel(name="unet", input_shape=(256, 256), channels=3,
                              n_class=2, blocks=4)
    dataset = SegmentationDataset(train.samples, model, data)
    image, mask = dataset[0]

    assert image.shape == (256, 256, 3)
    assert image.dtype == np.float32 and 0.0 <= image.min() and image.max() <= 1.0
    assert mask.shape == (256, 256, 2)
    assert np.array_equal(mask.sum(axis=-1), np.ones((256, 256), np.float32))
    # Nuclei are a minority of the picture but they are certainly there.
    foreground = mask[..., 1].mean()
    assert 0.0 < foreground < 0.5


def test_nine_different_source_sizes_all_normalise(train, data):
    model = SegmentationModel(name="unet", input_shape=(128, 128), n_class=2, blocks=4)
    dataset = SegmentationDataset(train.samples, model, data)
    shapes = {dataset[i][0].shape for i in range(0, len(dataset), 37)}
    assert shapes == {(128, 128, 3)}


def test_tiling_a_real_image_round_trips(train, data):
    """Cut a source image into 6 tiles and put it back exactly - the capability the old
    package was missing on the way out."""
    model = SegmentationModel(name="hd", input_shape=(256, 256), n_class=2, blocks=4,
                              splits=(2, 3))
    dataset = SegmentationDataset(train.samples, model, data)
    assert len(dataset) == 536 * 6

    image, _ = dataset._load(0)
    assert image.shape == (512, 768, 3)
    assert np.array_equal(stitch(tile(image, (2, 3)), (2, 3)), image)

    tile_image, tile_mask = dataset[0]
    assert tile_image.shape == (256, 256, 3)
    assert tile_mask.shape == (256, 256, 2)


def test_throughput_is_not_absurd(train, data):
    """Not a benchmark - a tripwire. If reading a sample ever takes a tenth of a second
    the GPU will sit idle and someone should notice here first.

    The **fastest** of three passes, not the only one. As a single pass this failed about
    one local run in eight - caught while a GPU run and two test loops were sharing the
    machine - and a tripwire that trips on a busy laptop teaches people to re-run it
    instead of reading it. The minimum is still a real measurement: a genuine regression
    makes every pass slow, and no amount of idle machine makes a slow reader fast.
    """
    model = SegmentationModel(name="unet", input_shape=(256, 256), n_class=2, blocks=4)
    dataset = SegmentationDataset(train.samples, model, data, cache_size=1)

    passes = []
    for _ in range(3):
        start = time.perf_counter()
        for index in range(30):
            dataset[index]
        passes.append((time.perf_counter() - start) / 30)

    per_sample = min(passes)
    assert per_sample < 0.1, (
        f"{per_sample * 1000:.0f} ms per sample is too slow "
        f"(passes: {', '.join(f'{p * 1000:.0f}' for p in passes)} ms)"
    )
