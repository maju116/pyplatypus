"""Samples on disk to training examples, for boxes.

The thing to test here is **correspondence, not shape**. Every array in this pipeline has
the right shape whether or not the boxes describe the objects: a target tensor is
(rows, cols, anchors, 5 + classes) no matter what is in it, and a letterbox applied to the
image and not to the boxes produces two perfectly well-formed arrays that disagree. So the
assertions below are about where things *are* - a box lands on its object, an inverse
returns what a forward took - which is the only kind of assertion that would have caught
the mask-resampling bug on the segmentation side.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyplatypus.data.detection import DetectionDataError, DetectionDataset
from pyplatypus.data.paths import discover_samples
from pyplatypus.spec.detection import DetectionData, DetectionModel

ANCHORS = (((0.35, 0.33), (0.25, 0.25)),
           ((0.18, 0.18), (0.12, 0.12)),
           ((0.08, 0.08), (0.05, 0.05)))


CLASSES = ["square", "bar"]


def build(root, *, model_kwargs=None, data_kwargs=None, only_images=False):
    data = DetectionData(train_path=str(root), validation_path=str(root),
                         classes=CLASSES, **(data_kwargs or {}))
    model = DetectionModel(name="d", input_shape=(128, 128), **(model_kwargs or {}))
    samples = discover_samples(root, subdirs=data.subdirs,
                               only_images=only_images).samples
    return DetectionDataset(samples, model, data, anchors=ANCHORS,
                            only_images=only_images)


# --- the letterbox is the whole point ---------------------------------------------------

def test_boxes_arrive_in_the_letterboxed_frame(tmp_path, voc_sample):
    """A 160-wide source into a 128-wide model: the scale is 0.8 and the padding is
    vertical, so a box at x=100 lands at x=80 and a box at y=0 lands at y=12.8."""
    voc_sample(tmp_path, "one", [(100, 0, 120, 20)], [0])
    example = build(tmp_path).read(0)

    assert example.fit.scale == pytest.approx(128 / 160)
    assert example.fit.pad_y == pytest.approx((128 - 128 * 0.8) / 2)
    assert example.fit.pad_x == pytest.approx(0.0)
    assert example.boxes[0] == pytest.approx([80.0, 12.8, 96.0, 28.8])
    assert example.image.shape == (128, 128, 3)


def test_the_inverse_puts_a_box_back_where_it_came_from(tmp_path, voc_sample):
    """What `predict` relies on. Asserted against the source coordinates rather than
    against a shape, because a box in the network's frame has the right shape too."""
    source = [(10, 12, 50, 60), (96, 20, 140, 34)]
    voc_sample(tmp_path, "one", source, [0, 1])
    example = build(tmp_path).read(0)

    # VOC is 1-based inclusive, so the box written as xmin=11 xmax=50 is 10..50 continuous.
    back = example.fit.inverse(example.boxes)
    assert back == pytest.approx(np.array(source, dtype=float))


def test_the_image_is_padded_rather_than_stretched(tmp_path, voc_sample):
    """Grey padding, and only where the aspect ratio needs it.

    The fill is 0.5 rather than 0: black is a legitimate pixel value in a radiograph, so
    padding with it invents tissue-free darkness that the model cannot tell from data.
    """
    voc_sample(tmp_path, "one", [(10, 10, 40, 40)], [0])
    image = build(tmp_path).read(0).image

    pad = int((128 - 128 * 0.8) / 2)
    assert image[:pad].min() == pytest.approx(0.5)
    assert image[-pad:].max() == pytest.approx(0.5)
    # The middle is the photograph, which is not all one value.
    assert image[pad + 2:-pad - 2].std() > 0.01


# --- the boxes label the pixels they are drawn on ---------------------------------------

def test_a_box_covers_the_object_it_labels(tmp_path, voc_sample):
    """The correspondence assertion. The fixture paints each box bright, so after
    letterboxing the pixels inside the transformed box must still be the bright ones - and
    a forward transform applied to the image but not the boxes, or to one axis only, fails
    this while every shape stays correct."""
    voc_sample(tmp_path, "one", [(20, 24, 60, 64)], [0])
    example = build(tmp_path).read(0)

    x0, y0, x1, y1 = example.boxes[0].astype(int)
    inside = example.image[y0 + 2:y1 - 2, x0 + 2:x1 - 2, 0]
    outside = example.image[2:y0 - 4, 2:x0 - 4, 0]
    assert inside.mean() > 0.7, "the box does not cover the object it labels"
    assert outside.mean() < 0.3


# --- targets ----------------------------------------------------------------------------

def test_one_sample_is_one_example_with_three_grids(tmp_path, voc_sample):
    voc_sample(tmp_path, "one", [(20, 24, 60, 64)], [0])
    image, targets = build(tmp_path)[0]

    assert image.shape == (128, 128, 3)
    assert len(targets) == 3
    rows = [target.shape[0] for target in targets]
    assert rows == [4, 8, 16], "128 over strides 32, 16, 8"
    for target in targets:
        # (rows, cols, anchors, 5 + classes)
        assert target.shape[2] == 2
        assert target.shape[3] == 5 + len(CLASSES)


def test_the_target_holds_the_object(tmp_path, voc_sample):
    """Objectness somewhere, and in exactly one slot: a box is placed once."""
    voc_sample(tmp_path, "one", [(20, 24, 60, 64)], [0])
    _, targets = build(tmp_path)[0]
    placed = sum(int((target[..., 4] > 0.5).sum()) for target in targets)
    assert placed == 1


def test_only_images_asks_for_no_annotation(tmp_path, voc_sample):
    """A test split may arrive without labels. The image still has to be letterboxed the
    same way, or predictions from it would be on a different frame."""
    voc_sample(tmp_path, "one", [(20, 24, 60, 64)], [0])
    (tmp_path / "one" / "annotations" / "one.xml").unlink()

    dataset = build(tmp_path, only_images=True)
    example = dataset.read(0)
    assert example.image.shape == (128, 128, 3)
    assert len(example.boxes) == 0
    assert example.fit.scale == pytest.approx(0.8)
    assert dataset[0][1] is None


# --- the survey -------------------------------------------------------------------------

def test_the_survey_counts_what_the_target_cannot_hold(tmp_path, voc_sample):
    """Two boxes of one shape whose centres fall in the same cell share a slot, so the
    second is dropped - never shown to the model and never counted as missed. Built here
    on purpose: at a 128 input the finest cell is 8 pixels, so two 24-pixel squares four
    pixels apart collide."""
    voc_sample(tmp_path, "crowded",
                     [(40, 40, 64, 64), (42, 42, 66, 66)], [0, 0])
    survey = build(tmp_path).survey()

    assert survey.images == 1
    assert survey.total == 2
    assert survey.unplaced == 1
    assert survey.unplaced_fraction == pytest.approx(0.5)


def test_the_survey_reads_no_pixels(tmp_path, voc_sample, monkeypatch):
    """It is meant to be cheap enough to run before every training run, which it only is
    if it never opens an image. Enforced rather than assumed."""
    voc_sample(tmp_path, "one", [(20, 24, 60, 64)], [0])
    dataset = build(tmp_path)

    import pyplatypus.data.detection as module

    def refuse(*args, **kwargs):
        raise AssertionError("the survey read an image")

    monkeypatch.setattr(module, "read_image", refuse)
    assert dataset.survey().total == 1


def test_a_degenerate_box_is_dropped_and_counted(tmp_path, voc_sample):
    """A click without a drag: `xmin == xmax`, which VOC's 1-based reading turns into one
    pixel. Two of BCCD's 4888 annotations are these."""
    voc_sample(tmp_path, "one", [(20, 24, 60, 64), (90, 90, 89, 89)], [0, 1])
    example = build(tmp_path).read(0)
    assert example.dropped == 1
    assert len(example.boxes) == 1


# --- refusals ---------------------------------------------------------------------------

def test_an_annotation_that_disagrees_with_its_image_is_refused(tmp_path, voc_sample):
    """The failure this exists for: a dataset is resized and the XML files are copied
    along unchanged. Every number stays plausible and every box lands somewhere else."""
    voc_sample(tmp_path, "one", [(20, 24, 60, 64)], [0],
                     declared_shape=(256, 320))
    with pytest.raises(DetectionDataError, match="320x256 and the file is 160x128"):
        build(tmp_path).read(0)


def test_several_annotation_files_for_one_sample_are_refused(tmp_path, voc_sample):
    """Segmentation joins several mask files into one. Boxes cannot be joined that way -
    two files are two opinions about one image, and nothing here can choose."""
    voc_sample(tmp_path, "one", [(20, 24, 60, 64)], [0])
    second = tmp_path / "one" / "annotations" / "another.xml"
    second.write_text((tmp_path / "one" / "annotations" / "one.xml").read_text())

    with pytest.raises(DetectionDataError, match="2 annotation files"):
        build(tmp_path).read(0)


def test_several_image_files_for_one_sample_are_refused(tmp_path, voc_sample):
    """Reading the first and ignoring the rest is the silent kind of wrong: the run
    trains, scores and reports, on part of the data, with nothing to say which part."""
    voc_sample(tmp_path, "one", [(20, 24, 60, 64)], [0])
    import shutil

    images = tmp_path / "one" / "images"
    shutil.copy(images / "one.png", images / "also.png")

    # At construction, not at first read: counting paths costs no I/O, and the
    # alternative is fitting anchors over the whole split and then failing on batch one.
    with pytest.raises(DetectionDataError, match="2 image files"):
        build(tmp_path)


def test_an_empty_dataset_is_refused(tmp_path):
    model = DetectionModel(name="d", input_shape=(128, 128))
    data = DetectionData(train_path=".", validation_path=".", classes=CLASSES)
    with pytest.raises(DetectionDataError, match="at least one sample"):
        DetectionDataset((), model, data, anchors=ANCHORS)


# --- the other annotation format --------------------------------------------------------

def test_labelme_json_reads_the_same_way(tmp_path, voc_sample):
    """One pipeline, two formats. The boxes have to land in the same place, which is the
    claim worth checking - not that the reader returns something."""
    source = [(10, 12, 50, 60)]
    voc_sample(tmp_path, "one", source, [0])
    (tmp_path / "one" / "annotations" / "one.xml").unlink()
    height, width = (128, 160)
    (tmp_path / "one" / "annotations" / "one.json").write_text(json.dumps({
        "imageHeight": height, "imageWidth": width, "imagePath": "one.png",
        "shapes": [{"label": "square", "shape_type": "rectangle",
                    "points": [[10, 12], [50, 60]]}],
    }))

    example = build(tmp_path, data_kwargs={"annotation_format": "labelme"}).read(0)
    assert example.fit.inverse(example.boxes) == pytest.approx(
        np.array(source, dtype=float)
    )


def test_the_coordinate_convention_moves_every_box(tmp_path, voc_sample):
    """A pixel and a shrink, on every box. Worth a test because it is invisible by eye and
    costs real IoU on a small object: at 24 pixels a side, one pixel is 4%."""
    voc_sample(tmp_path, "one", [(10, 12, 50, 60)], [0])

    as_voc = build(tmp_path).read(0)
    as_zero = build(tmp_path, data_kwargs={"coordinates": "zero_based"}).read(0)

    voc_box = as_voc.fit.inverse(as_voc.boxes)[0]
    zero_box = as_zero.fit.inverse(as_zero.boxes)[0]
    assert voc_box == pytest.approx([10.0, 12.0, 50.0, 60.0])
    assert zero_box == pytest.approx([11.0, 13.0, 50.0, 60.0])
    assert (zero_box[2] - zero_box[0]) < (voc_box[2] - voc_box[0])
