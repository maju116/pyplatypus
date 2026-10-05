"""Reading volumes, and the corrections without which an array means nothing.

The test that earns its keep is `test_two_files_describing_the_same_anatomy_read_alike`.
Everything else here checks shapes and refusals; that one checks the claim the module is
for.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyplatypus.data.volumes import (
    VolumeError,
    crop_or_pad,
    looks_like_volume,
    read_volume,
    resample_to_spacing,
    resize_volume,
    volume_spacing,
)

nib = pytest.importorskip("nibabel")


def write_volume(path, array, affine=None, spacing=(1.0, 1.0, 1.0)):
    if affine is None:
        affine = np.diag([*spacing, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine), str(path))
    return path


def hounsfield_block(shape=(6, 8, 4)):
    """Something CT-shaped: air, soft tissue and bone in known places."""
    volume = np.full(shape, -1000.0, dtype=np.float32)   # air
    volume[1:4, 2:6, 1:3] = 40.0                          # soft tissue
    volume[2, 3, 2] = 1200.0                              # a spot of bone
    return volume


# ----------------------------------------------------------------- recognising
def test_volume_files_are_recognised_by_name():
    assert looks_like_volume("brain.nii")
    assert looks_like_volume("/data/BRAIN.NII.GZ")
    assert not looks_like_volume("slice.png")
    assert not looks_like_volume("scan.dcm")


# ---------------------------------------------------------------- orientation
def test_two_files_describing_the_same_anatomy_read_alike(tmp_path):
    """The reason `as_closest_canonical` is not optional.

    A NIfTI's affine says where the voxels are; the array order alone does not. Written one
    way and then again with the first axis reversed - array flipped and affine flipped to
    match, so both files describe *the same patient* - a naive reader returns mirror images.
    Train on both and the model can learn which dataset a scan came from instead of the
    anatomy, and nothing in either file looks wrong.
    """
    volume = hounsfield_block()
    forwards = write_volume(tmp_path / "forwards.nii.gz", volume,
                            affine=np.diag([1.0, 1.0, 1.0, 1.0]))

    flipped_affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    flipped_affine[0, 3] = volume.shape[0] - 1        # same origin after the flip
    backwards = write_volume(tmp_path / "backwards.nii.gz", volume[::-1],
                             affine=flipped_affine)

    a = read_volume(forwards, window="soft_tissue")
    b = read_volume(backwards, window="soft_tissue")
    assert np.array_equal(a, b)


def test_spacing_comes_back_in_the_order_the_reader_uses(tmp_path):
    path = write_volume(tmp_path / "scan.nii.gz", hounsfield_block(),
                        spacing=(0.7, 0.7, 5.0))
    # 0.7 x 0.7 in plane and 5 mm between slices: the ordinary shape of a clinical CT, and
    # the reason anything reporting a volume in millilitres needs this.
    assert volume_spacing(path) == pytest.approx((0.7, 0.7, 5.0))


def test_spacing_is_positive_even_when_the_affine_is_not(tmp_path):
    affine = np.diag([-1.5, -2.0, 3.0, 1.0])
    path = write_volume(tmp_path / "scan.nii.gz", hounsfield_block(), affine=affine)
    assert all(z > 0 for z in volume_spacing(path))


# ------------------------------------------------------------------ intensity
def test_a_named_window_maps_the_same_numbers_the_same_way(tmp_path):
    """The point of Hounsfield units: 'lung' means one thing in every scan.

    Checked by scaling one voxel far out of range in the second file. A per-volume
    minimum-maximum would move everything else; a fixed window leaves it alone.
    """
    volume = hounsfield_block()
    plain = write_volume(tmp_path / "a.nii.gz", volume)

    with_implant = volume.copy()
    with_implant[0, 0, 0] = 30000.0      # metal, or a corrupted voxel
    spiked = write_volume(tmp_path / "b.nii.gz", with_implant)

    a = read_volume(plain, window="lung")
    b = read_volume(spiked, window="lung")
    assert np.array_equal(a[1:], b[1:])


def test_a_window_taken_from_the_data_does_move_with_an_outlier(tmp_path):
    """The counterexample, so the difference is on the record rather than asserted."""
    volume = hounsfield_block()
    plain = write_volume(tmp_path / "a.nii.gz", volume)
    spiked_array = volume.copy()
    spiked_array[0, 0, 0] = 30000.0
    spiked = write_volume(tmp_path / "b.nii.gz", spiked_array)

    a = read_volume(plain, window="auto")
    b = read_volume(spiked, window="auto")
    assert not np.array_equal(a[1:], b[1:])


def test_an_explicit_centre_and_width_works(tmp_path):
    path = write_volume(tmp_path / "scan.nii.gz", hounsfield_block())
    values = read_volume(path, window=(40.0, 400.0))
    assert 0.0 <= float(values.min()) and float(values.max()) <= 1.0


def test_an_unknown_window_lists_the_ones_there_are(tmp_path):
    path = write_volume(tmp_path / "scan.nii.gz", hounsfield_block())
    with pytest.raises(VolumeError, match="unknown window"):
        read_volume(path, window="pancreas_ish")


def test_a_window_of_no_width_is_refused(tmp_path):
    path = write_volume(tmp_path / "scan.nii.gz", hounsfield_block())
    with pytest.raises(VolumeError, match="width must be positive"):
        read_volume(path, window=(40.0, 0.0))


def test_a_label_map_keeps_its_integers(tmp_path):
    """Read with `nearest`, labels must survive untouched.

    Dividing a label map by its maximum turns class 1 of 2 into 0.5, and every voxel of
    that class then belongs to no class at all.
    """
    labels = np.zeros((4, 4, 2), dtype=np.float32)
    labels[1:3, 1:3, :] = 1
    labels[2, 2, 0] = 2
    path = write_volume(tmp_path / "labels.nii.gz", labels)

    read = read_volume(path, nearest=True)
    assert sorted(np.unique(read).tolist()) == [0.0, 1.0, 2.0]


# -------------------------------------------------------------------- shapes
def test_a_volume_arrives_channels_last(tmp_path):
    path = write_volume(tmp_path / "scan.nii.gz", hounsfield_block((6, 8, 4)))
    assert read_volume(path).shape == (6, 8, 4, 1)


def test_channels_can_be_repeated_for_a_three_channel_model(tmp_path):
    path = write_volume(tmp_path / "scan.nii.gz", hounsfield_block())
    assert read_volume(path, channels=3).shape == (6, 8, 4, 3)


def test_resizing_happens_on_request(tmp_path):
    path = write_volume(tmp_path / "scan.nii.gz", hounsfield_block((6, 8, 4)))
    assert read_volume(path, size=(4, 4, 2)).shape == (4, 4, 2, 1)


def test_resizing_a_label_map_invents_no_new_labels():
    labels = np.zeros((8, 8, 4, 1), dtype=np.float32)
    labels[2:6, 2:6, 1:3] = 2.0
    resized = resize_volume(labels, (4, 4, 2), nearest=True)
    assert sorted(np.unique(resized).tolist()) == [0.0, 2.0]


def test_resizing_an_image_does_interpolate():
    values = np.zeros((8, 8, 4, 1), dtype=np.float32)
    values[4:, :, :] = 1.0
    resized = resize_volume(values, (5, 8, 4))
    assert 0.0 < float(resized.min()) or len(np.unique(resized)) > 2


def test_resizing_to_the_same_size_is_a_no_op():
    values = np.zeros((4, 4, 2, 1), dtype=np.float32)
    assert resize_volume(values, (4, 4, 2)) is values


def test_a_wrong_number_of_sizes_is_refused():
    values = np.zeros((4, 4, 2, 1), dtype=np.float32)
    with pytest.raises(VolumeError, match="three sizes"):
        resize_volume(values, (4, 4))


def test_a_2d_array_is_not_a_volume():
    with pytest.raises(VolumeError, match="channels-last volume"):
        resize_volume(np.zeros((4, 4, 1), dtype=np.float32), (2, 2, 2))


def test_a_missing_file_says_which(tmp_path):
    with pytest.raises(VolumeError, match="could not read"):
        read_volume(tmp_path / "absent.nii.gz")


def test_a_file_that_is_not_a_volume_is_reported_as_one(tmp_path):
    """The narrowed exception list, exercised. A PNG renamed to .nii.gz reaches nibabel and
    comes back as a VolumeError rather than as whatever nibabel felt like raising."""
    impostor = tmp_path / "not-really.nii.gz"
    impostor.write_bytes(b"this is not a NIfTI file")
    with pytest.raises(VolumeError, match="could not read"):
        read_volume(impostor)


def test_a_truncated_volume_is_reported_as_one(tmp_path):
    path = write_volume(tmp_path / "scan.nii.gz", hounsfield_block())
    content = path.read_bytes()
    path.write_bytes(content[: len(content) // 3])
    with pytest.raises(VolumeError, match="could not read"):
        read_volume(path)


# ------------------------------------------------------------------ resampling
def sphere(shape, spacing, radius_mm=10.0):
    """A ball of fixed physical radius, sampled on whatever grid is asked for.

    The point of the fixture: the same anatomy, acquired two ways. Voxel counts differ, the
    millimetres do not.
    """
    grid = np.indices(shape).astype(np.float32)
    centre = (np.asarray(shape, dtype=np.float32) - 1) / 2
    millimetres = [(grid[axis] - centre[axis]) * spacing[axis] for axis in range(3)]
    distance = np.sqrt(sum(axis ** 2 for axis in millimetres))
    return np.where(distance <= radius_mm, 40.0, -1000.0).astype(np.float32)


def test_resampling_brings_two_acquisitions_onto_one_scale(tmp_path):
    """The claim the feature exists for, measured rather than asserted loosely.

    The same 10 mm ball, acquired at 1 mm slices and at 2.5 mm. Counted in voxels as they
    arrive, the coarse scan holds 1688 where the fine one holds 4224 - a 60% difference in
    what is physically the same object. Resampled to a common voxel size it reads 3760, an 11%
    difference, and the remaining gap is honest: at 2.5 mm the caps of the ball were never
    measured, and no resampling recovers information an acquisition did not take.

    So the test compares the error before and after, which is the actual promise. Numbers for
    a 10 mm sphere: 4189 mm3 in theory, 4224 measured at 1 mm.
    """
    fine = write_volume(tmp_path / "fine.nii.gz", sphere((40, 40, 40), (1.0, 1.0, 1.0)),
                        spacing=(1.0, 1.0, 1.0))
    coarse = write_volume(tmp_path / "coarse.nii.gz", sphere((40, 40, 16), (1.0, 1.0, 2.5)),
                          spacing=(1.0, 1.0, 2.5))

    reference = read_volume(fine, window="soft_tissue")
    native = read_volume(coarse, window="soft_tissue")
    resampled = resample_to_spacing(native, volume_spacing(coarse), (1.0, 1.0, 1.0))

    def voxels(volume):
        return int((volume > 0.4).sum())

    # Non-empty first: `approx(0, rel=...)` accepts zero, and the first version of this test
    # compared two empty counts and passed while proving nothing.
    assert voxels(reference) > 1000
    assert voxels(native) > 500
    assert voxels(resampled) > 1000

    before = abs(voxels(native) - voxels(reference)) / voxels(reference)
    after = abs(voxels(resampled) - voxels(reference)) / voxels(reference)
    assert before > 0.4          # the problem, as it arrives
    assert after < 0.15          # the same object, now on the same scale
    assert after < before / 3


def test_without_resampling_the_same_anatomy_comes_out_different_sizes(tmp_path):
    """The counterexample, so the difference is on the record rather than asserted."""
    fine = write_volume(tmp_path / "fine.nii.gz", sphere((40, 40, 40), (1.0, 1.0, 1.0)),
                        spacing=(1.0, 1.0, 1.0))
    coarse = write_volume(tmp_path / "coarse.nii.gz", sphere((40, 40, 16), (1.0, 1.0, 2.5)),
                          spacing=(1.0, 1.0, 2.5))

    # Both resized into the same box, which is the pipeline's behaviour without a target
    # spacing.
    a = read_volume(fine, window="soft_tissue", size=(40, 40, 32))
    b = read_volume(coarse, window="soft_tissue", size=(40, 40, 32))
    voxels_a = int((a > 0.4).sum())
    voxels_b = int((b > 0.4).sum())
    assert voxels_a > 1000 and voxels_b > 1000
    # The 2.5 mm scan covers 40 mm of anatomy where the 1 mm one covers 40 slices of 1 mm, so
    # stretched into the same box the ball fills a different fraction of it.
    assert voxels_b != pytest.approx(voxels_a, rel=0.1)


def test_resampling_computes_the_shape_from_the_ratio():
    volume = np.zeros((10, 10, 40, 1), dtype=np.float32)
    assert resample_to_spacing(volume, (1, 1, 2.5), (1, 1, 1)).shape[:3] == (10, 10, 100)
    assert resample_to_spacing(volume, (1, 1, 1), (2, 2, 2)).shape[:3] == (5, 5, 20)


def test_resampling_never_produces_an_empty_axis():
    # A very thin volume asked for very coarse voxels still has to have one slice.
    volume = np.zeros((4, 4, 2, 1), dtype=np.float32)
    assert resample_to_spacing(volume, (1, 1, 1), (10, 10, 10)).shape[:3] == (1, 1, 1)


def test_resampling_a_label_map_invents_no_labels():
    labels = np.zeros((8, 8, 8, 1), dtype=np.float32)
    labels[2:6, 2:6, 2:6] = 2.0
    resampled = resample_to_spacing(labels, (1, 1, 1), (0.5, 0.5, 0.5), nearest=True)
    assert sorted(np.unique(resampled).tolist()) == [0.0, 2.0]


def test_bad_spacing_is_refused():
    volume = np.zeros((4, 4, 4, 1), dtype=np.float32)
    with pytest.raises(VolumeError, match="three numbers"):
        resample_to_spacing(volume, (1, 1), (1, 1, 1))
    with pytest.raises(VolumeError, match="must be positive"):
        resample_to_spacing(volume, (1, 1, 0), (1, 1, 1))


# ---------------------------------------------------------------- crop and pad
def test_cropping_takes_the_middle():
    volume = np.arange(8 * 4 * 2, dtype=np.float32).reshape(8, 4, 2, 1)
    cropped = crop_or_pad(volume, (4, 4, 2))
    assert cropped.shape == (4, 4, 2, 1)
    # Centred: rows 2..5 of eight.
    assert np.array_equal(cropped, volume[2:6])


def test_padding_puts_the_volume_in_the_middle():
    volume = np.ones((2, 2, 2, 1), dtype=np.float32)
    padded = crop_or_pad(volume, (4, 4, 4))
    assert padded.shape == (4, 4, 4, 1)
    assert np.array_equal(padded[1:3, 1:3, 1:3], volume)
    assert float(padded[0, 0, 0, 0]) == 0.0


def test_cropping_and_padding_can_happen_on_different_axes():
    volume = np.ones((8, 2, 4, 1), dtype=np.float32)
    assert crop_or_pad(volume, (4, 6, 4)).shape == (4, 6, 4, 1)


def test_the_pad_value_can_be_chosen():
    volume = np.ones((2, 2, 2, 1), dtype=np.float32)
    padded = crop_or_pad(volume, (4, 2, 2), pad_value=0.25)
    assert float(padded[0, 0, 0, 0]) == 0.25


def test_fitting_an_array_of_the_wrong_rank_is_refused():
    with pytest.raises(VolumeError, match="spatial axes plus channels"):
        crop_or_pad(np.zeros((4, 4, 1), dtype=np.float32), (2, 2, 2))


# ------------------------------------------------------- through the pipeline
def write_case(root, name, shape, spacing, radius_mm=10.0):
    """One case: a ball in air, and a label map marking it."""
    sample = root / name
    (sample / "images").mkdir(parents=True, exist_ok=True)
    (sample / "masks").mkdir(parents=True, exist_ok=True)
    scan = sphere(shape, spacing, radius_mm)
    labels = (scan > 0).astype(np.float32)
    write_volume(sample / "images" / "ct.nii.gz", scan, spacing=spacing)
    write_volume(sample / "masks" / "seg.nii.gz", labels, spacing=spacing)
    return sample


def test_the_pipeline_resamples_both_image_and_mask_the_same_way(tmp_path):
    """Two cases covering different amounts of the patient, one target spacing, one shape.

    The fixtures matter here and the first version of this test got them wrong. Two scans at
    different slice *thickness* that cover the same extent survive a plain resize unharmed -
    the relative scale is preserved. What breaks it is a different **field of view**: 40 slices
    of 1 mm covers 40 mm of the patient and 40 slices of 2.5 mm covers 100 mm, so squeezing
    both into one box makes the same ball a different size. That is the ordinary state of
    clinical data, where the scanned length depends on the question asked.

    Both come out at the model's shape either way; with resampling the anatomy is at a common
    scale, and the mask has to follow the image exactly or every labelled voxel is misplaced.
    """
    from pyplatypus.data.dataset import SegmentationDataset
    from pyplatypus.data.paths import discover_samples
    from pyplatypus.spec.data import SegmentationData
    from pyplatypus.spec.models import SegmentationModel

    root = tmp_path / "cases"
    write_case(root, "fine", (40, 40, 40), (1.0, 1.0, 1.0))      # 40 mm of patient
    write_case(root, "coarse", (40, 40, 40), (1.0, 1.0, 2.5))    # 100 mm of patient

    data = SegmentationData(
        train_path=str(root), validation_path=str(root), labels=[0, 1],
        window="soft_tissue", target_spacing=(1.0, 1.0, 1.0),
    )
    model = SegmentationModel(name="m", input_shape=(32, 32, 32), channels=1, 
                              blocks=2)
    dataset = SegmentationDataset(discover_samples(root).samples, model, data)

    shapes, foreground = [], []
    for index in range(len(dataset)):
        image, mask = dataset[index]
        shapes.append(image.shape)
        assert mask.shape == (32, 32, 32, 2)
        # The mask must mark the ball, not a translated copy of it: where the image is tissue,
        # the mask's foreground channel has to be the one that is set.
        tissue = image[..., 0] > 0.4
        assert float(mask[..., 1][tissue].mean()) > 0.9
        foreground.append(int(mask[..., 1].sum()))

    assert shapes == [(32, 32, 32, 1), (32, 32, 32, 1)]
    # Same anatomy at a common voxel size: the labelled volume agrees to within what the
    # coarser acquisition could see.
    assert foreground[0] == pytest.approx(foreground[1], rel=0.15)


def test_without_a_target_spacing_the_two_fields_of_view_disagree(tmp_path):
    """The same two cases, resized instead of resampled - the behaviour before this existed.

    One scan covering 40 mm and one covering 100 mm, both squeezed into a 32-voxel box: the
    same ball ends up more than twice the size in one of them, and nothing in the data says so.
    """
    from pyplatypus.data.dataset import SegmentationDataset
    from pyplatypus.data.paths import discover_samples
    from pyplatypus.spec.data import SegmentationData
    from pyplatypus.spec.models import SegmentationModel

    root = tmp_path / "cases"
    write_case(root, "fine", (40, 40, 40), (1.0, 1.0, 1.0))      # 40 mm of patient
    write_case(root, "coarse", (40, 40, 40), (1.0, 1.0, 2.5))    # 100 mm of patient

    data = SegmentationData(train_path=str(root), validation_path=str(root), labels=[0, 1],
                            window="soft_tissue")
    model = SegmentationModel(name="m", input_shape=(32, 32, 32), channels=1, 
                              blocks=2)
    dataset = SegmentationDataset(discover_samples(root).samples, model, data)

    foreground = [int(dataset[i][1][..., 1].sum()) for i in range(len(dataset))]
    assert foreground[0] != pytest.approx(foreground[1], rel=0.15)


def test_padding_a_label_map_adds_background_not_a_new_class(tmp_path):
    """A volume smaller than the input shape is padded, and the padding has to be background.

    Padded with zero, which is outside any label list that does not contain it - and
    `labels_to_classes` maps anything unmatched to class 0. Either way the padding is class 0,
    which is the background by convention.
    """
    from pyplatypus.data.dataset import SegmentationDataset
    from pyplatypus.data.paths import discover_samples
    from pyplatypus.spec.data import SegmentationData
    from pyplatypus.spec.models import SegmentationModel

    root = tmp_path / "cases"
    write_case(root, "small", (16, 16, 8), (1.0, 1.0, 1.0), radius_mm=4.0)

    data = SegmentationData(train_path=str(root), validation_path=str(root), labels=[0, 1],
                            window="soft_tissue", target_spacing=(1.0, 1.0, 1.0))
    model = SegmentationModel(name="m", input_shape=(32, 32, 32), channels=1, 
                              blocks=2)
    dataset = SegmentationDataset(discover_samples(root).samples, model, data)

    image, mask = dataset[0]
    assert image.shape == (32, 32, 32, 1)
    # The corners are padding: background class, and nothing else.
    assert float(mask[0, 0, 0, 0]) == 1.0
    assert float(mask[0, 0, 0, 1]) == 0.0
    assert mask.shape[-1] == 2
