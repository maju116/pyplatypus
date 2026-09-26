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
    looks_like_volume,
    read_volume,
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
