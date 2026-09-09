import numpy as np
import pytest

from pyplatypus.data import SegmentationDataset, build_augmenter, discover
from pyplatypus.spec.components import AugmentationStep
from pyplatypus.spec.models import SegmentationModel


def make_model(**overrides):
    base = {"name": "m", "input_shape": (64, 64), "channels": 3, "n_class": 2, "blocks": 2}
    return SegmentationModel(**{**base, **overrides})


@pytest.fixture
def dataset(nested_root, binary_data):
    samples = discover(nested_root, binary_data).samples
    return SegmentationDataset(samples, make_model(), binary_data)


def test_length_is_samples_when_not_tiling(dataset):
    assert len(dataset) == 3


def test_shapes_are_channels_last(dataset):
    image, mask = dataset[0]
    assert image.shape == (64, 64, 3)
    assert mask.shape == (64, 64, 2)


def test_images_arrive_scaled_to_unit_range(dataset):
    image, _ = dataset[0]
    assert image.dtype == np.float32
    assert 0.0 <= image.min() and image.max() <= 1.0


def test_mask_is_one_hot(dataset):
    _, mask = dataset[0]
    assert np.array_equal(mask.sum(axis=-1), np.ones((64, 64), np.float32))


def test_tiling_multiplies_the_examples(nested_root, binary_data):
    """One source image becomes four training examples, and every one is a full tile."""
    samples = discover(nested_root, binary_data).samples
    model = make_model(splits=(2, 2))
    data = SegmentationDataset(samples, model, binary_data)
    assert len(data) == 3 * 4
    image, mask = data[0]
    assert image.shape == (64, 64, 3)      # the tile is input_shape
    assert mask.shape == (64, 64, 2)


def test_tiles_of_one_sample_are_consecutive(nested_root, binary_data):
    samples = discover(nested_root, binary_data).samples
    data = SegmentationDataset(samples, make_model(splits=(2, 2)), binary_data)
    assert len(data) == 12
    data[0], data[3]        # same source sample
    assert len(data._cache) == 1


def test_only_images_yields_no_mask(nested_root, binary_data):
    samples = discover(nested_root, binary_data, only_images=True).samples
    data = SegmentationDataset(samples, make_model(), binary_data, only_images=True)
    image, mask = data[0]
    assert mask is None and image.shape == (64, 64, 3)


def test_out_of_range_index(dataset):
    with pytest.raises(IndexError):
        dataset[999]


def test_negative_index_counts_from_the_end(dataset):
    assert np.array_equal(dataset[-1][0], dataset[len(dataset) - 1][0])


def test_augmentation_moves_labels_not_colours(nested_root, binary_data):
    """Masks go through augmentation as class indices, so a flip cannot blend a label
    into a colour that belongs to no class."""
    samples = discover(nested_root, binary_data).samples
    augmenter = build_augmenter([AugmentationStep(name="HorizontalFlip", params={"p": 1.0})])
    data = SegmentationDataset(samples, make_model(), binary_data, augmenter=augmenter)
    _, mask = data[0]
    assert set(np.unique(mask)) <= {0.0, 1.0}


def test_colormap_coverage_flags_a_wrong_colormap(nested_root, binary_data):
    samples = discover(nested_root, binary_data).samples
    wrong = binary_data.model_copy(update={"colormap": [(1, 2, 3), (4, 5, 6)]})
    data = SegmentationDataset(samples, make_model(), wrong)
    assert data.colormap_coverage() > 0.9

    right = SegmentationDataset(samples, make_model(), binary_data)
    assert right.colormap_coverage() == pytest.approx(0.0)


def test_3d_model_is_refused_by_the_2d_pipeline(nested_root, binary_data):
    from pyplatypus.data.dataset import DataError

    samples = discover(nested_root, binary_data).samples
    volume = make_model(input_shape=(64, 64, 64))
    with pytest.raises(DataError, match="3D"):
        SegmentationDataset(samples, volume, binary_data)
