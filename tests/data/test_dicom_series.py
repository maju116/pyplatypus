"""Assembling a folder of slices into a volume.

Every test here corresponds to a way of getting it wrong that produces a volume which trains
without complaint: the wrong order, two series stacked together, a missing slice, or a window
resolved per slice instead of per series.
"""

from __future__ import annotations

import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file

from pyplatypus.data.dicom import DicomError
from pyplatypus.data.dicom_series import (
    describe_series,
    looks_like_dicom_series,
    read_dicom_series,
    series_spacing,
)
from pyplatypus.data.images import read_image

CT = get_testdata_file("CT_small.dcm")


def write_series(directory, positions, *, names=None, uid=None, spacing=(0.8, 0.8),
                 orientation=(1, 0, 0, 0, 1, 0), fill=None, pixels=None, window=None,
                 instance_numbers=None, thickness=2.5, origin=(0.0, 0.0)):
    """A series on disk, with the geometry the test cares about and nothing else.

    Names default to something whose lexicographic order disagrees with the anatomy, because
    a reader that sorts by filename has to fail these tests rather than pass them by luck.
    """
    directory.mkdir(parents=True, exist_ok=True)
    uid = uid or pydicom.uid.generate_uid()
    if names is None:
        names = [f"IM{(len(positions) - i) * 7 % 100:02d}.dcm" for i in range(len(positions))]

    written = []
    for index, (position, name) in enumerate(zip(positions, names, strict=True)):
        dataset = pydicom.dcmread(CT)
        dataset.SeriesInstanceUID = uid
        dataset.ImageOrientationPatient = list(orientation)
        dataset.ImagePositionPatient = [float(origin[0]), float(origin[1]), float(position)]
        dataset.PixelSpacing = list(spacing)
        dataset.SliceThickness = thickness
        if instance_numbers is not None:
            dataset.InstanceNumber = instance_numbers[index]
        if window is not None:
            dataset.WindowCenter, dataset.WindowWidth = window[index]
        if fill is not None:
            dataset.PixelData = np.full_like(dataset.pixel_array, fill[index]).tobytes()
        if pixels is not None:
            plane = np.asarray(pixels[index], dtype=dataset.pixel_array.dtype)
            dataset.PixelData = np.ascontiguousarray(plane).tobytes()
        path = directory / name
        dataset.save_as(path)
        written.append(path)
    return written


# ---------------------------------------------------------------- recognising
def test_a_directory_of_slices_is_recognised(tmp_path):
    write_series(tmp_path / "series", [0.0, 2.5, 5.0])
    assert looks_like_dicom_series(tmp_path / "series")


def test_one_slice_is_not_a_series(tmp_path):
    write_series(tmp_path / "single", [0.0])
    assert not looks_like_dicom_series(tmp_path / "single")


def test_a_file_is_not_a_series(tmp_path):
    assert not looks_like_dicom_series(CT)


def test_a_directory_of_something_else_is_not_a_series(tmp_path):
    other = tmp_path / "pictures"
    other.mkdir()
    (other / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    (other / "b.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    assert not looks_like_dicom_series(other)


# --------------------------------------------------------------------- order
def test_slices_are_ordered_by_position_not_by_filename(tmp_path):
    """The mistake that costs nothing to make and everything to find.

    The files are named so that sorting them as text gives a different order from the
    anatomy. Each slice is filled with a distinct value, so the assembled volume says
    plainly which order it was stacked in.
    """
    root = tmp_path / "series"
    write_series(
        root,
        positions=[0.0, 2.5, 5.0, 7.5],
        names=["z.dcm", "a.dcm", "m.dcm", "b.dcm"],
        fill=[10, 20, 30, 40],
    )
    volume = read_dicom_series(root, window="full")

    # Increasing position means increasing fill: 10, 20, 30, 40 along the slice axis.
    middles = [float(volume[64, 64, k, 0]) for k in range(4)]
    assert middles == sorted(middles)
    assert len(set(middles)) == 4


def test_the_order_survives_the_files_being_listed_backwards(tmp_path):
    root = tmp_path / "series"
    write_series(root, positions=[7.5, 5.0, 2.5, 0.0], fill=[40, 30, 20, 10])
    volume = read_dicom_series(root, window="full")
    middles = [float(volume[64, 64, k, 0]) for k in range(4)]
    assert middles == sorted(middles)


def test_instance_number_is_the_fallback_when_geometry_is_missing(tmp_path):
    root = tmp_path / "series"
    paths = write_series(root, positions=[0.0, 2.5, 5.0], instance_numbers=[1, 2, 3])
    for path in paths:
        dataset = pydicom.dcmread(str(path))
        del dataset.ImagePositionPatient
        dataset.save_as(path)

    series = describe_series(root)
    assert series.sorted_by == "instance_number"
    assert [int(pydicom.dcmread(str(p)).InstanceNumber) for p in series.paths] == [1, 2, 3]


def test_without_geometry_or_instance_numbers_it_refuses_to_guess(tmp_path):
    root = tmp_path / "series"
    paths = write_series(root, positions=[0.0, 2.5, 5.0])
    for path in paths:
        dataset = pydicom.dcmread(str(path))
        del dataset.ImagePositionPatient
        del dataset.InstanceNumber
        dataset.save_as(path)

    with pytest.raises(DicomError, match="no way to tell what order"):
        describe_series(root)


# -------------------------------------------------------------------- series
def test_two_series_in_one_directory_are_refused(tmp_path):
    """A folder out of an archive usually holds several: a scout, a reconstruction, a phase.
    Stacked together they interleave two anatomies at two resolutions."""
    root = tmp_path / "mixed"
    write_series(root, positions=[0.0, 2.5], names=["a1.dcm", "a2.dcm"])
    write_series(root, positions=[0.0, 2.5], names=["b1.dcm", "b2.dcm"])

    with pytest.raises(DicomError, match="different series"):
        describe_series(root)


def test_slices_of_different_sizes_are_refused(tmp_path):
    root = tmp_path / "series"
    paths = write_series(root, positions=[0.0, 2.5, 5.0])
    dataset = pydicom.dcmread(str(paths[1]))
    dataset.Rows = int(dataset.Rows) // 2
    dataset.save_as(paths[1])

    with pytest.raises(DicomError, match="not all the same size"):
        describe_series(root)


def test_slices_at_different_angles_are_refused(tmp_path):
    root = tmp_path / "series"
    paths = write_series(root, positions=[0.0, 2.5, 5.0])
    dataset = pydicom.dcmread(str(paths[2]))
    dataset.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
    dataset.save_as(paths[2])

    with pytest.raises(DicomError, match="oriented differently"):
        describe_series(root)


def test_different_pixel_spacing_is_refused(tmp_path):
    root = tmp_path / "series"
    paths = write_series(root, positions=[0.0, 2.5, 5.0])
    dataset = pydicom.dcmread(str(paths[0]))
    dataset.PixelSpacing = [0.5, 0.5]
    dataset.save_as(paths[0])

    with pytest.raises(DicomError, match="different PixelSpacing"):
        describe_series(root)


# ----------------------------------------------------------------------- gaps
def test_a_missing_slice_is_an_error_not_a_shorter_volume(tmp_path):
    """The one that would never show up in a metric.

    A volume stacked over a gap does not lose a slice, it puts everything past the gap in the
    wrong place - and every score computed on it looks perfectly ordinary.
    """
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5, 7.5, 10.0])     # 5.0 is missing

    with pytest.raises(DicomError, match="not evenly spaced"):
        describe_series(root)


def test_the_message_says_where_the_gap_is(tmp_path):
    # Long enough for the median gap to mean something, which is what lets the message name
    # the offender.
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5, 5.0, 10.0, 12.5, 15.0],
                 names=["a.dcm", "b.dcm", "c.dcm", "d.dcm", "e.dcm", "f.dcm"])
    with pytest.raises(DicomError, match="after 'c.dcm'"):
        describe_series(root)


def test_with_too_few_slices_it_says_it_cannot_tell_which_gap_is_wrong(tmp_path):
    """Two gaps and the median sits between them. Naming one would be arithmetic accident
    dressed up as a diagnosis."""
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5, 7.5], names=["a.dcm", "b.dcm", "c.dcm"])
    with pytest.raises(DicomError, match="no way to tell which gap"):
        describe_series(root)


def test_two_slices_in_the_same_place_are_refused(tmp_path):
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5, 2.5, 5.0])
    with pytest.raises(DicomError, match="same position"):
        describe_series(root)


def test_slight_scanner_jitter_is_tolerated(tmp_path):
    # Positions come with a few decimals of noise; refusing on that would reject most real
    # series while catching nothing.
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5001, 4.9998, 7.5002])
    series = describe_series(root)
    assert series.spacing[2] == pytest.approx(2.5, abs=1e-3)


# ---------------------------------------------------------------- one window
def test_one_window_is_used_for_the_whole_series(tmp_path):
    """Slices of one series can record different windows.

    Honoured slice by slice, the same tissue is a different brightness on adjacent slices - a
    gradient the scanner never measured. The window is resolved once, from the first slice.
    """
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5, 5.0], fill=[500, 500, 500],
                 window=[(40, 400), (1000, 4000), (-600, 1500)])

    volume = read_dicom_series(root, window="auto")
    slices = [float(volume[64, 64, k, 0]) for k in range(3)]
    # Same stored value everywhere, so one window means one brightness.
    assert slices[0] == pytest.approx(slices[1]) == pytest.approx(slices[2])


def test_a_named_window_beats_whatever_the_files_recorded(tmp_path):
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5], window=[(1000, 4000), (1000, 4000)])
    named = read_dicom_series(root, window="lung")
    recorded = read_dicom_series(root, window="auto")
    assert not np.allclose(named, recorded)


# --------------------------------------------------------------- the geometry
def test_slice_spacing_is_measured_not_read_from_thickness(tmp_path):
    """SliceThickness says how thick a slice is, not how far apart they sit. For an
    overlapping reconstruction those differ, and using thickness puts every slice but the
    first in the wrong place."""
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 1.25, 2.5, 3.75], thickness=2.5)  # 50% overlap
    series = describe_series(root)
    assert series.spacing[2] == pytest.approx(1.25)


def test_spacing_comes_back_in_the_canonical_order(tmp_path):
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5, 5.0], spacing=(0.7, 0.7))
    assert series_spacing(root) == pytest.approx((0.7, 0.7, 2.5), abs=1e-4)


def test_the_same_anatomy_stored_two_ways_reads_alike(tmp_path):
    """The claim the NIfTI reader makes, for the same reason and with the same test shape.

    One series is stored with the ordinary in-plane axes. The other stores *the same patient*
    with both in-plane directions reversed - which means the pixel rows and columns are
    flipped and the origin moves to the opposite corner. A reader that stacked arrays without
    reading the geometry would return one of them mirrored, and nothing in either file would
    look wrong.
    """
    pattern = np.zeros((128, 128), dtype=np.int16)
    pattern[10:40, 20:60] = 400          # asymmetric, so a flip cannot hide

    plain = tmp_path / "plain"
    write_series(plain, positions=[0.0, 2.5, 5.0], orientation=(1, 0, 0, 0, 1, 0),
                 pixels=[pattern, pattern, pattern], spacing=(1.0, 1.0))

    # Reversed in-plane axes: increasing column index now runs in -x, increasing row index in
    # -y, so the stored array is flipped both ways and the origin is the far corner.
    flipped = tmp_path / "flipped"
    write_series(flipped, positions=[0.0, 2.5, 5.0], orientation=(-1, 0, 0, 0, -1, 0),
                 pixels=[pattern[::-1, ::-1].copy()] * 3, spacing=(1.0, 1.0),
                 origin=(127.0, 127.0))

    a = read_dicom_series(plain, window="full")
    b = read_dicom_series(flipped, window="full")
    assert np.array_equal(a, b)


def test_the_volume_arrives_channels_last_and_scaled(tmp_path):
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5, 5.0])
    volume = read_dicom_series(root, window="soft_tissue")
    assert volume.shape == (128, 128, 3, 1)
    assert 0.0 <= float(volume.min()) and float(volume.max()) <= 1.0


def test_channels_can_be_repeated(tmp_path):
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5])
    assert read_dicom_series(root, channels=3).shape[-1] == 3


def test_resizing_happens_on_request(tmp_path):
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5, 5.0, 7.5])
    assert read_dicom_series(root, size=(32, 32, 2)).shape == (32, 32, 2, 1)


def test_read_image_dispatches_a_directory_to_the_series_reader(tmp_path):
    """So a CSV can name a series directory in the same column that names a file."""
    root = tmp_path / "series"
    write_series(root, positions=[0.0, 2.5, 5.0])
    volume = read_image(root, channels=1, size=(32, 32, 3))
    assert volume.shape == (32, 32, 3, 1)


def test_a_directory_with_no_dicom_in_it_says_so(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DicomError, match="no DICOM files"):
        describe_series(empty)


def test_describe_series_reads_no_pixels(tmp_path):
    """Checking a hundred cases should not cost a hundred gigabytes, so the checks run on
    headers alone. Pinned by breaking the pixel data and describing the series anyway."""
    root = tmp_path / "series"
    paths = write_series(root, positions=[0.0, 2.5, 5.0])
    dataset = pydicom.dcmread(str(paths[1]))
    dataset.PixelData = b"\x00" * 8            # far too short to decode
    dataset.save_as(paths[1])

    series = describe_series(root)             # headers only: fine
    assert len(series) == 3
    with pytest.raises(DicomError):            # pixels: not fine
        read_dicom_series(root)


def test_a_series_per_case_trains_end_to_end(tmp_path):
    """The layout data actually arrives in: the scan as a folder of slices, the segmentation
    as a NIfTI beside it. Nothing in the spec mentions series - a 3D model and several DICOM
    files in one sample is enough to say what they are."""
    nib = pytest.importorskip("nibabel")
    from pyplatypus import Engine, from_dict

    root = tmp_path / "cases"
    for case in range(3):
        sample = root / f"case_{case:02d}"
        write_series(sample / "images", positions=[0.0, 2.5, 5.0, 7.5],
                     spacing=(1.0, 1.0))
        labels = np.zeros((128, 128, 4), dtype=np.float32)
        labels[20:60, 20:60, 1:3] = 1
        (sample / "masks").mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(labels, np.diag([1.0, 1.0, 2.5, 1.0])),
                 str(sample / "masks" / "seg.nii.gz"))

    spec = from_dict({
        "task": "semantic_segmentation",
        "data": {"train_path": str(root), "validation_path": str(root),
                 "labels": [0, 1], "window": "soft_tissue", "shuffle": False},
        "models": [{"name": "unet3d", "input_shape": [32, 32, 4], 
                    "channels": 1, "blocks": 2, "filters": 4, "batch_size": 1,
                    "epochs": 1, "metrics": [{"name": "dice"}]}],
    })
    engine = Engine(spec, device="cpu")
    history = engine.fit()["unet3d"]

    assert len(history) == 1
    masks = engine.predict("unet3d", split="validation")
    assert masks.shape == (3, 32, 32, 4, 2)


def test_several_volumes_per_sample_need_their_order_stated(tmp_path):
    """Four NIfTI files per case is a multi-modal dataset - BraTS shape. Reading them as
    channels works, but only once `channels_from` says which file is which: sorted names give
    an order that is reproducible and anatomically meaningless. Without it, refused."""
    nib = pytest.importorskip("nibabel")
    from pyplatypus.data.dataset import DataError, SegmentationDataset
    from pyplatypus.data.paths import discover_samples
    from pyplatypus.spec.data import SegmentationData
    from pyplatypus.spec.models import SegmentationModel

    root = tmp_path / "cases"
    sample = root / "case_00"
    (sample / "images").mkdir(parents=True, exist_ok=True)
    (sample / "masks").mkdir(parents=True, exist_ok=True)
    for modality in ("t1", "t2"):
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 4), np.float32), np.eye(4)),
                 str(sample / "images" / f"{modality}.nii.gz"))
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 4), np.float32), np.eye(4)),
             str(sample / "masks" / "seg.nii.gz"))

    data = SegmentationData(train_path=str(root), validation_path=str(root), labels=[0, 1])
    model = SegmentationModel(name="m", input_shape=(8, 8, 4), channels=1, 
                              blocks=2)
    dataset = SegmentationDataset(discover_samples(root).samples, model, data)
    with pytest.raises(DataError, match="channels_from"):
        dataset[0]
