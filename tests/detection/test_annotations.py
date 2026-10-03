"""Reading boxes off disk, and the one-pixel convention that decides their area."""

import json

import pytest

from pyplatypus.detection.annotations import (
    describe_annotations,
    read_annotations,
    read_labelme,
    read_voc,
)
from pyplatypus.detection.metrics import DetectionError

LABELS = ["RBC", "WBC", "Platelets"]


def voc_file(path, objects, *, width=640, height=480):
    body = "".join(
        f"<object><name>{name}</name><difficult>{int(hard)}</difficult><bndbox>"
        f"<xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax>"
        f"</bndbox></object>"
        for name, (x1, y1, x2, y2), hard in objects
    )
    path.write_text(
        f"<annotation><filename>{path.stem}.jpg</filename>"
        f"<size><width>{width}</width><height>{height}</height></size>{body}</annotation>"
    )
    return path


# --- the one-pixel question -------------------------------------------------------------

def test_voc_indices_become_the_right_width(tmp_path):
    """Pascal VOC stores 1-based inclusive indices, so pixels 1 to 10 is ten pixels wide.
    Reading `xmax - xmin` gives nine, and the box's area is 81% of the truth."""
    file = voc_file(tmp_path / "a.xml", [("RBC", (1, 1, 10, 10), False)])
    corrected = read_voc(file, LABELS).boxes[0]
    naive = read_voc(file, LABELS, coordinates="zero_based").boxes[0]

    assert corrected.tolist() == [0.0, 0.0, 10.0, 10.0]
    assert corrected[2] - corrected[0] == 10.0
    assert naive[2] - naive[0] == 9.0
    # The size of the mistake, stated as a number rather than a worry.
    assert ((naive[2] - naive[0]) / (corrected[2] - corrected[0])) ** 2 == pytest.approx(0.81)


def test_a_zero_minimum_proves_the_file_is_not_one_based(tmp_path):
    """And is refused rather than turned into -1, with the alternative named."""
    file = voc_file(tmp_path / "a.xml", [("RBC", (0, 5, 10, 15), False)])
    with pytest.raises(DetectionError, match="coordinates='zero_based'"):
        read_voc(file, LABELS)
    # Which works, and leaves the numbers alone.
    assert read_voc(file, LABELS, coordinates="zero_based").boxes[0].tolist() == \
        [0.0, 5.0, 10.0, 15.0]


def test_describe_reports_the_evidence_without_converting(tmp_path):
    voc_file(tmp_path / "a.xml", [("RBC", (0, 5, 10, 15), False)])
    voc_file(tmp_path / "b.xml", [("WBC", (20, 20, 60, 60), False)])
    report = describe_annotations(tmp_path, LABELS)

    assert report["files"] == 2
    assert report["objects"] == 2
    assert report["per_class"] == {"RBC": 1, "WBC": 1, "Platelets": 0}
    assert report["minimum_coordinate"] == 0.0
    # The only part that is proof: a 1-based index cannot be zero.
    assert report["coordinates"] == "zero_based"


def test_describe_does_not_claim_proof_it_does_not_have(tmp_path):
    """No zero minimum is consistent with 1-based and does not establish it, so the
    wording has to stop short of saying so."""
    voc_file(tmp_path / "a.xml", [("RBC", (5, 5, 10, 15), False)])
    report = describe_annotations(tmp_path, LABELS)
    assert "consistent with" in report["coordinates"]


# --- reading ---------------------------------------------------------------------------

def test_a_voc_file_reads_its_objects(tmp_path):
    file = voc_file(tmp_path / "a.xml", [
        ("RBC", (10, 20, 50, 60), False),
        ("WBC", (100, 100, 200, 200), True),
    ])
    annotation = read_voc(file, LABELS)

    assert (annotation.width, annotation.height) == (640, 480)
    assert annotation.labels.tolist() == [0, 1]
    assert annotation.names == ["RBC", "WBC"]
    assert annotation.difficult.tolist() == [False, True]
    assert annotation.image_path == "a.jpg"


def test_difficult_reaches_the_metric_unchanged(tmp_path):
    """`as_truth()` is the handover to `detection_report`, which excludes difficult
    objects as Pascal VOC's own evaluation excludes them."""
    file = voc_file(tmp_path / "a.xml", [("RBC", (1, 1, 10, 10), True)])
    truth = read_voc(file, LABELS).as_truth()
    assert truth["difficult"].tolist() == [True]


def test_an_unknown_class_is_refused_by_default(tmp_path):
    file = voc_file(tmp_path / "a.xml", [("Lymphocyte", (1, 1, 10, 10), False)])
    with pytest.raises(DetectionError, match="not in the label list"):
        read_voc(file, LABELS)


def test_an_unknown_class_can_be_skipped_on_purpose(tmp_path):
    """Which is how you train on two of a dataset's three classes."""
    file = voc_file(tmp_path / "a.xml", [
        ("Lymphocyte", (1, 1, 10, 10), False),
        ("RBC", (20, 20, 30, 30), False),
    ])
    annotation = read_voc(file, LABELS, strict_labels=False)
    assert annotation.labels.tolist() == [0]


def test_an_annotation_with_no_objects_is_a_result_not_an_error(tmp_path):
    """An image with nothing in it is data, and a detector has to be scored on it."""
    file = voc_file(tmp_path / "a.xml", [])
    annotation = read_voc(file, LABELS)
    assert annotation.boxes.shape == (0, 4)
    assert annotation.labels.shape == (0,)


@pytest.mark.parametrize("broken, match", [
    ("<annotation><object/></annotation>", "no <size>"),
    ("<annotation><size><width>10</width></size></annotation>", "no <height>"),
    ("not xml at all", "not readable XML"),
])
def test_a_malformed_file_says_what_is_wrong(tmp_path, broken, match):
    file = tmp_path / "a.xml"
    file.write_text(broken)
    with pytest.raises(DetectionError, match=match):
        read_voc(file, LABELS)


def test_a_non_numeric_coordinate_names_the_tag(tmp_path):
    file = tmp_path / "a.xml"
    file.write_text(
        "<annotation><size><width>10</width><height>10</height></size>"
        "<object><name>RBC</name><bndbox><xmin>one</xmin><ymin>1</ymin>"
        "<xmax>5</xmax><ymax>5</ymax></bndbox></object></annotation>"
    )
    with pytest.raises(DetectionError, match="xmin"):
        read_voc(file, LABELS)


# --- LabelMe ---------------------------------------------------------------------------

def labelme_file(path, shapes, *, width=640, height=480):
    path.write_text(json.dumps({
        "imagePath": f"{path.stem}.jpg",
        "imageWidth": width, "imageHeight": height,
        "shapes": shapes,
    }))
    return path


def test_a_labelme_rectangle_reads_in_either_corner_order(tmp_path):
    """LabelMe gives two opposite corners in no guaranteed order."""
    forwards = labelme_file(tmp_path / "a.json", [
        {"label": "WBC", "points": [[10, 20], [50, 60]], "shape_type": "rectangle"}])
    backwards = labelme_file(tmp_path / "b.json", [
        {"label": "WBC", "points": [[50, 60], [10, 20]], "shape_type": "rectangle"}])

    assert read_labelme(forwards, LABELS).boxes[0].tolist() == [10, 20, 50, 60]
    assert read_labelme(backwards, LABELS).boxes[0].tolist() == [10, 20, 50, 60]


def test_a_polygon_becomes_its_bounding_box(tmp_path):
    """Which is the only reading detection can use, and is a loss: the shape goes."""
    file = labelme_file(tmp_path / "a.json", [
        {"label": "RBC", "points": [[10, 10], [30, 5], [40, 25], [15, 30]],
         "shape_type": "polygon"}])
    assert read_labelme(file, LABELS).boxes[0].tolist() == [10, 5, 40, 30]


def test_labelme_coordinates_are_left_alone(tmp_path):
    """They are already continuous, so the VOC correction would be wrong here."""
    file = labelme_file(tmp_path / "a.json", [
        {"label": "RBC", "points": [[1, 1], [10, 10]], "shape_type": "rectangle"}])
    box = read_labelme(file, LABELS).boxes[0]
    assert box.tolist() == [1, 1, 10, 10]


def test_a_labelme_shape_with_one_point_is_refused(tmp_path):
    file = labelme_file(tmp_path / "a.json", [
        {"label": "RBC", "points": [[10, 10]], "shape_type": "point"}])
    with pytest.raises(DetectionError, match="at least two"):
        read_labelme(file, LABELS)


def test_labelme_without_a_frame_is_refused(tmp_path):
    file = tmp_path / "a.json"
    file.write_text(json.dumps({"shapes": []}))
    with pytest.raises(DetectionError, match="no imageHeight"):
        read_labelme(file, LABELS)


# --- a directory -----------------------------------------------------------------------

def test_a_directory_reads_in_sorted_order(tmp_path):
    """Sorted, so a split taken from the order is reproducible."""
    for name in ("c", "a", "b"):
        voc_file(tmp_path / f"{name}.xml", [("RBC", (1, 1, 10, 10), False)])
    names = [a.path.stem for a in read_annotations(tmp_path, LABELS)]
    assert names == ["a", "b", "c"]


def test_an_empty_directory_says_which_format_it_looked_for(tmp_path):
    with pytest.raises(DetectionError, match="annotation_format='labelme'"):
        read_annotations(tmp_path, LABELS)


def test_the_labelme_reader_is_reachable_from_the_directory_reader(tmp_path):
    labelme_file(tmp_path / "a.json", [
        {"label": "RBC", "points": [[1, 1], [10, 10]], "shape_type": "rectangle"}])
    out = read_annotations(tmp_path, LABELS, annotation_format="labelme")
    assert len(out) == 1 and out[0].labels.tolist() == [0]
