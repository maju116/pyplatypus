import numpy as np
import pytest

from pyplatypus.data import SegmentationDataset, build_augmenter, discover
from pyplatypus.spec.components import AugmentationStep
from pyplatypus.spec.models import SegmentationModel


def make_model(**overrides):
    base = {"name": "m", "input_shape": (64, 64), "channels": 3, "blocks": 2}
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
    assert image.shape == (64, 64, 3)  # the tile is input_shape
    assert mask.shape == (64, 64, 2)


def test_tiles_of_one_sample_are_consecutive(nested_root, binary_data):
    samples = discover(nested_root, binary_data).samples
    data = SegmentationDataset(samples, make_model(splits=(2, 2)), binary_data)
    assert len(data) == 12
    data[0], data[3]  # same source sample
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


def test_the_pipeline_serves_volumes(volume_root, volume_data):
    """What used to be a scope line. The same dataset class, one rank up: the model asks for
    a 3D input_shape and the samples are NIfTI files with label maps for masks."""
    pytest.importorskip("nibabel")
    samples = discover(volume_root, volume_data).samples
    model = make_model(input_shape=(8, 8, 4))
    dataset = SegmentationDataset(samples, model, volume_data)

    image, mask = dataset[0]
    assert image.shape == (8, 8, 4, 3)
    assert mask.shape == (8, 8, 4, 2)
    assert mask.sum(axis=-1).min() == 1.0  # every voxel belongs to exactly one class


def test_volume_patches_come_out_of_the_same_tiling(volume_root, volume_data):
    """`splits` in 3D is patch sampling, and it is the same code as the 2D grid."""
    pytest.importorskip("nibabel")
    samples = discover(volume_root, volume_data).samples
    # (2, 2, 1) over an 8x8x4 volume gives four 4x4x4 patches. The depth is not cut because
    # four slices are already the smallest the network can pool twice.
    model = make_model(input_shape=(4, 4, 4), splits=(2, 2, 1))
    dataset = SegmentationDataset(samples, model, volume_data)

    assert len(dataset) == len(samples) * 4
    image, mask = dataset[0]
    assert image.shape == (4, 4, 4, 3)
    assert mask.shape == (4, 4, 4, 2)


# --- a wide label map through the whole reader --------------------------------------------


@pytest.fixture
def wide_label_root(tmp_path):
    """One sample whose mask is a 16-bit label map holding 0, 255 and 1000.

    The combination that made `pyplatypus#346` silent: 255 is one of the labels and a
    higher value is another, so the clip sent both to one class with nothing unmatched.
    """
    from PIL import Image

    root = tmp_path / "data" / "s0"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(root / "images" / "i.png")
    values = np.zeros((64, 64), np.uint16)
    values[:20] = 255
    values[20:40] = 1000
    Image.fromarray(values).save(root / "masks" / "m.png")
    return tmp_path / "data"


def test_a_16_bit_label_map_reaches_the_target_as_three_classes(wide_label_root):
    from pyplatypus.spec.data import SegmentationData

    data = SegmentationData(
        train_path=str(wide_label_root),
        validation_path=str(wide_label_root),
        labels=[0, 255, 1000],
    )
    samples = discover(wide_label_root, data).samples
    dataset = SegmentationDataset(samples, make_model(), data)

    _, mask = dataset[0]
    assert mask.shape == (64, 64, 3)
    # Each of the three bands is one class, and the third is the one the clip destroyed.
    assert mask[:20, :, 1].all() and mask[20:40, :, 2].all() and mask[40:, :, 0].all()


def test_the_mask_check_sees_a_class_above_255(wide_label_root):
    """The loud half, which was misdiagnosed rather than silent: a label the file does hold
    on every row of a band was reported as never appearing, and the refusal blamed the
    specification for it."""
    from pyplatypus.spec.data import SegmentationData

    data = SegmentationData(
        train_path=str(wide_label_root),
        validation_path=str(wide_label_root),
        labels=[0, 255, 1000],
    )
    samples = discover(wide_label_root, data).samples
    report = SegmentationDataset(samples, make_model(), data).inspect_masks()

    assert report.unmatched == 0.0
    assert report.present_classes == [0, 1, 2]
    assert report.missing_classes == []
