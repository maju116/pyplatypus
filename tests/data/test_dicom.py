"""Reading DICOM.

Opening the file is the easy part. These tests are about the parts that are invisible
when they are wrong: stored values that were never converted to real units, a window that
depends on the image it is applied to, and an inverted photometric interpretation.
"""

import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file
from pydicom.pixels import apply_modality_lut

from pyplatypus.data.dicom import DicomError, looks_like_dicom, read_dicom, resolve_window
from pyplatypus.data.images import read_image
from pyplatypus.spec.common import WINDOWS

CT = get_testdata_file("CT_small.dcm")  # rescale, no window tags
MR = get_testdata_file("MR_small.dcm")  # window tags, no rescale
RGB = get_testdata_file("SC_rgb.dcm")  # three channels, 8-bit


def test_a_dicom_is_recognised_by_its_contents():
    # Exports out of an archive often have no extension, or one the archive invented, so
    # trusting the name would refuse perfectly good files.
    assert looks_like_dicom(CT)


def test_something_that_is_not_a_dicom_is_not_mistaken_for_one(tmp_path):
    ordinary = tmp_path / "picture.png"
    ordinary.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    assert not looks_like_dicom(ordinary)
    assert not looks_like_dicom(tmp_path / "missing.dcm.gone")


def test_the_extension_is_enough_when_it_is_there(tmp_path):
    named = tmp_path / "scan.dcm"
    named.write_bytes(b"")
    assert looks_like_dicom(named)


def test_stored_values_become_real_units():
    """The step the package this replaces skipped entirely.

    CT stores arbitrary integers; `stored * RescaleSlope + RescaleIntercept` turns them
    into Hounsfield units, where air is -1000 and water 0 on every scanner ever built.
    Without it a model trained on one machine means nothing on another, and nothing in the
    data says so.
    """
    dataset = pydicom.dcmread(CT)
    stored = dataset.pixel_array
    real = apply_modality_lut(stored, dataset)
    assert float(real.min()) == pytest.approx(float(stored.min()) + dataset.RescaleIntercept)
    assert dataset.RescaleIntercept != 0  # this file would show the difference


def test_a_fixed_window_does_not_let_one_pixel_rescale_the_image():
    """The reason a window is fixed rather than taken from the image.

    Scaling by the image's own extremes is the obvious choice and the wrong one: a metal
    implant, a marker or an artefact rescales everything else. Here one bright pixel moves
    the mean of the rest of the image by nearly half.
    """
    dataset = pydicom.dcmread(CT)
    values = np.asarray(apply_modality_lut(dataset.pixel_array, dataset), dtype=np.float32)
    with_artefact = values.copy()
    with_artefact[0, 0] = 3000.0

    by_extremes = lambda a: (a - a.min()) / (a.max() - a.min())
    low, high = (
        WINDOWS["soft_tissue"][0] - WINDOWS["soft_tissue"][1] / 2,
        WINDOWS["soft_tissue"][0] + WINDOWS["soft_tissue"][1] / 2,
    )
    by_window = lambda a: np.clip((a - low) / (high - low), 0, 1)

    assert not np.allclose(by_extremes(values)[1:], by_extremes(with_artefact)[1:])
    assert np.allclose(by_window(values)[1:], by_window(with_artefact)[1:])


def test_monochrome1_is_inverted():
    """Low values are bright in MONOCHROME1. Left alone, every such image trains as its
    own negative, which looks like nothing being wrong."""
    dataset = pydicom.dcmread(CT)
    normal = read_dicom(CT, window="full")

    inverted_path = str(CT)
    dataset.PhotometricInterpretation = "MONOCHROME1"
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        inverted_path = pathlib.Path(directory) / "inverted.dcm"
        dataset.save_as(inverted_path)
        inverted = read_dicom(inverted_path, window="full")
    assert np.allclose(inverted, 1.0 - normal, atol=1e-5)


@pytest.mark.parametrize(
    "path,channels,shape",
    [
        (CT, 1, (128, 128, 1)),
        (MR, 1, (64, 64, 1)),
        (RGB, 3, (100, 100, 3)),
    ],
)
def test_modalities_read_to_a_unit_range(path, channels, shape):
    array = read_dicom(path, channels=channels)
    assert array.shape == shape
    assert array.dtype == np.float32
    assert 0.0 <= array.min() and array.max() <= 1.0


def test_a_named_window_is_the_same_as_its_numbers():
    assert np.allclose(read_dicom(CT, window="soft_tissue"), read_dicom(CT, window=(40, 400)))


def test_different_windows_show_different_things():
    lung = read_dicom(CT, window="lung")
    bone = read_dicom(CT, window="bone")
    assert not np.allclose(lung, bone)
    # Lung windows are centred on air, so most tissue saturates bright.
    assert lung.mean() > bone.mean()


def test_an_unknown_window_lists_the_real_ones():
    with pytest.raises(DicomError, match="soft_tissue"):
        read_dicom(CT, window="sof_tissue")


def test_a_window_of_no_width_is_refused():
    with pytest.raises(DicomError, match="positive"):
        read_dicom(CT, window=(40, 0))


def test_auto_uses_the_window_in_the_file():
    dataset = pydicom.dcmread(MR)
    assert resolve_window(dataset, "auto") is not None  # this file has the tags
    assert resolve_window(pydicom.dcmread(CT), "auto") is None  # this one does not


def test_channels_are_converted_rather_than_refused():
    assert read_dicom(RGB, channels=1).shape == (100, 100, 1)
    assert read_dicom(CT, channels=3).shape == (128, 128, 3)


def test_read_image_dispatches_without_being_told():
    direct = read_dicom(CT, channels=1)
    through = read_image(CT, channels=1)
    assert np.allclose(direct, through)


def test_resizing_keeps_the_precision_the_rescale_recovered():
    # Through the 8-bit path a Hounsfield range would collapse to 256 levels, undoing
    # exactly what the modality LUT was applied for.
    full = read_image(CT, channels=1)
    smaller = read_image(CT, channels=1, size=(64, 64))
    assert smaller.shape == (64, 64, 1)
    assert smaller.dtype == np.float32
    assert len(np.unique(smaller)) > 256 or len(np.unique(full)) <= 256


def test_a_corrupt_file_says_which_one():
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        broken = pathlib.Path(directory) / "broken.dcm"
        broken.write_bytes(b"\x00" * 200)
        with pytest.raises(DicomError, match="broken.dcm"):
            read_dicom(broken)


def dicom_dataset(root, n=4, size=64):
    """A dataset of DICOM images with PNG masks - the mixed arrangement a real archive
    tends to produce, where the pixels come from the scanner and the labels from whoever
    drew them."""
    import pathlib

    from PIL import Image

    dataset = pydicom.dcmread(CT)
    rng = np.random.default_rng(0)
    for index in range(n):
        sample = pathlib.Path(root) / f"case_{index:02d}"
        (sample / "images").mkdir(parents=True, exist_ok=True)
        (sample / "masks").mkdir(parents=True, exist_ok=True)

        top = 8 + index * 4
        stored = np.full((size, size), 0, dtype=np.int16)
        stored[top : top + 20, 10:40] = 1200  # a bright structure to find
        stored = stored + rng.integers(0, 60, (size, size)).astype(np.int16)

        slice_ds = dataset.copy()
        slice_ds.Rows, slice_ds.Columns = size, size
        slice_ds.PixelData = stored.tobytes()
        slice_ds.save_as(sample / "images" / "scan.dcm")

        mask = np.zeros((size, size), np.uint8)
        mask[top : top + 20, 10:40] = 255
        Image.fromarray(mask).convert("RGB").save(sample / "masks" / "label.png")


def test_a_dicom_dataset_trains_end_to_end(tmp_path):
    """The point of all of the above: pixels out of a scanner, through the same pipeline,
    into a model, with no conversion step asked of the user."""
    from pyplatypus import Engine, from_dict

    dicom_dataset(tmp_path)
    spec = from_dict(
        {
            "task": "semantic_segmentation",
            "data": {
                "train_path": str(tmp_path),
                "validation_path": str(tmp_path),
                "colormap": [[0, 0, 0], [255, 255, 255]],
                "dicom_window": "soft_tissue",
            },
            "models": [
                {
                    "name": "ct",
                    "input_shape": [32, 32],
                    "channels": 1,
                    "blocks": 2,
                    "filters": 4,
                    "epochs": 2,
                    "batch_size": 2,
                    "loss": {"name": "cce_dice"},
                    "metrics": [{"name": "dice", "include_background": False}],
                }
            ],
        }
    )

    engine = Engine(spec, device="cpu")
    histories = engine.fit()
    assert len(histories["ct"]) == 2

    table = engine.evaluate()
    assert table[0]["model"] == "ct"
    assert 0.0 <= table[0]["dice"] <= 1.0

    masks = engine.predict("ct", split="validation")
    assert masks.shape == (4, 32, 32, 2)


def test_the_window_reaches_the_pipeline(tmp_path):
    """A setting nothing acts on is worse than no setting: it reads as a decision the
    user made and the software ignored."""
    from pyplatypus.data import SegmentationDataset, discover
    from pyplatypus.spec.data import SegmentationData
    from pyplatypus.spec.models import SegmentationModel

    dicom_dataset(tmp_path, n=2)
    model = SegmentationModel(name="m", input_shape=(32, 32), channels=1, blocks=2)

    def first_image(window):
        data = SegmentationData(
            train_path=str(tmp_path),
            validation_path=str(tmp_path),
            colormap=[(0, 0, 0), (255, 255, 255)],
            dicom_window=window,
        )
        samples = discover(tmp_path, data).samples
        return SegmentationDataset(samples, model, data)[0][0]

    assert not np.allclose(first_image("lung"), first_image("bone"))
