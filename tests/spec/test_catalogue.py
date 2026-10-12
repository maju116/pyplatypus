"""The listings answer with what the engine accepts, not with a list somebody maintained.

Every assertion below reads the schema or the models and compares the listing against
*that*, never against a literal list of names - because a literal list is the thing this
module exists to remove. The one hand-written mapping it does contain, the per-architecture
notes, is here because the architectures are an enum with no docstring per member, and it is
held by a completeness test rather than by discipline.
"""

import inspect

import pytest

from pyplatypus import (
    available_callbacks,
    available_losses,
    available_metrics,
    available_models,
    available_optimizers,
    available_tasks,
    spec_schema,
)
from pyplatypus.detection_engine import DetectionReport
from pyplatypus.errors import PlatypusError
from pyplatypus.spec.catalogue import _ARCHITECTURE_NOTES
from pyplatypus.spec.common import Architecture, Task

SCHEMA = spec_schema()
DEFS = SCHEMA["$defs"]


def union_names(model: str, field: str) -> set[str]:
    """The names a discriminated union accepts, read out of the schema.

    Read from the `discriminator.mapping` pydantic emits, which is the authoritative answer
    to the question a user is really asking - *what may I write here* - and the one the
    documentation site publishes. A listing that disagrees with it is wrong whatever the
    models say. For a list-valued field the union sits under `items`.
    """
    body = DEFS[model]["properties"][field]
    body = body.get("items", body)
    return set(body["discriminator"]["mapping"])


@pytest.mark.parametrize(
    ("listing", "model", "field"),
    [
        (available_losses, "SegmentationModel", "loss"),
        (available_metrics, "SegmentationModel", "metrics"),
        (available_optimizers, "SegmentationModel", "optimizer"),
        (available_callbacks, "SegmentationModel", "callbacks"),
    ],
    ids=["losses", "metrics", "optimizers", "callbacks"],
)
def test_a_listing_names_exactly_what_the_schema_accepts(listing, model, field):
    assert {row["name"] for row in listing()} == union_names(model, field)


def test_the_listings_are_not_trivially_short():
    """Or every comparison above could pass on an empty set meeting an empty set, which is
    §4's "two numbers that were both zero" in its usual costume."""
    assert len(available_losses()) == 13
    assert len(available_metrics()) == 4
    assert len(available_optimizers()) == 8
    assert len(available_callbacks()) == 7
    assert len(available_models()) == 4
    assert len(available_tasks()) == 2


def test_architectures_come_from_the_enum_and_every_member_has_a_note():
    """The hand-written half, gated. A fifth architecture added to the enum fails here
    rather than appearing in the listing with an empty column."""
    assert {row["name"] for row in available_models()} == {m.value for m in Architecture}
    assert set(_ARCHITECTURE_NOTES) == set(Architecture)
    assert all(len(note) > 120 for note in _ARCHITECTURE_NOTES.values())


def test_tasks_come_from_the_task_enum():
    assert {row["name"] for row in available_tasks()} == {member.value for member in Task}


@pytest.mark.parametrize(
    ("listing", "definition_of"),
    [
        (available_losses, {"dice": "DiceLoss", "boundary": "BoundaryLoss"}),
        (available_metrics, {"cldice": "ClDiceMetric", "iou": "IouMetric"}),
        (available_optimizers, {"adam": "Adam", "sgd": "Sgd"}),
        (available_callbacks, {"swa": "Swa", "csv_logger": "CsvLogger"}),
    ],
    ids=["losses", "metrics", "optimizers", "callbacks"],
)
def test_the_description_is_the_schema_description_shortened(listing, definition_of):
    """The listing must not acquire prose of its own: what it shows is the component's
    first paragraph, so the page, the schema and this table cannot disagree."""
    rows = {row["name"]: row for row in listing()}
    for name, definition in definition_of.items():
        published = DEFS[definition]["description"]
        assert rows[name]["description"] == " ".join(published.split("\n\n")[0].split())
        # Not a length floor: a summary line is allowed to be short, and `dice`'s is 48
        # characters. The floor on the whole description lives in
        # test_every_component_is_described.py, where it belongs - here the claim is only
        # that the column is the published summary and not empty.
        assert rows[name]["description"]
        assert "\n" not in rows[name]["description"]


@pytest.mark.parametrize(
    ("listing", "definition_of"),
    [
        (available_losses, {"tversky": "TverskyLoss", "lovasz": "LovaszLoss"}),
        (available_metrics, {"tversky": "TverskyMetric"}),
        (available_optimizers, {"sgd": "Sgd", "adadelta": "Adadelta"}),
        (available_callbacks, {"early_stopping": "EarlyStopping"}),
    ],
    ids=["losses", "metrics", "optimizers", "callbacks"],
)
def test_the_arguments_are_the_fields_the_component_takes(listing, definition_of):
    rows = {row["name"]: row for row in listing()}
    for name, definition in definition_of.items():
        properties = set(DEFS[definition]["properties"]) - {"name"}
        assert set(rows[name]["arguments"]) == properties


def test_detection_chooses_no_loss_component_and_says_what_it_has_instead():
    """An empty table would be the third answer and the wrong one, so this refuses - and the
    refusal names the two values of `box_loss` read from the schema, so it cannot go stale
    while the schema moves."""
    offered = DEFS["DetectionModel"]["properties"]["box_loss"]["enum"]
    assert len(offered) == 2
    with pytest.raises(PlatypusError) as raised:
        available_losses(Task.OBJECT_DETECTION)
    for value in offered:
        assert str(value) in str(raised.value)
    assert "box_loss" in str(raised.value)


def test_detection_chooses_no_metrics_and_names_the_ones_it_reports():
    """The three names in that message are keys of the detection report, and this is what
    ties the two together: rename a key and the message stops being true here rather than in
    somebody's console."""
    source = inspect.getsource(DetectionReport.as_row)
    with pytest.raises(PlatypusError) as raised:
        available_metrics("object_detection")
    message = str(raised.value)
    for key in ("map_50", "map_50_95", "mean_matched_iou"):
        assert f'"{key}"' in source, f"{key} is no longer a column of the detection report"
        assert key in message
    assert "report" in message


def test_an_unknown_task_lists_the_ones_there_are():
    with pytest.raises(PlatypusError) as raised:
        available_losses("instance_segmentation")
    for member in Task:
        assert member.value in str(raised.value)


def test_detection_models_list_the_detector_and_its_own_arguments():
    rows = available_models("object_detection")
    assert [row["name"] for row in rows] == ["yolo3"]
    properties = set(DEFS["DetectionModel"]["properties"]) - {"name", "architecture"}
    assert set(rows[0]["arguments"]) == properties
