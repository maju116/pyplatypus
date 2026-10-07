"""The JSON Schema is a published description of what a configuration may contain, so it
has to be real and it has to be stable."""

import json

from pyplatypus import spec_schema, write_schema


def test_schema_is_json_serialisable():
    json.dumps(spec_schema())


def test_the_top_level_is_a_choice_of_task():
    """Since detection arrived there is no single top-level shape: `task` selects one, and
    the schema has to say so in the way a validator can act on, not only in prose."""
    schema = spec_schema()
    assert schema["$id"].endswith("spec.schema.json")
    assert schema["discriminator"] == {
        "propertyName": "task",
        "mapping": {
            "semantic_segmentation": "#/$defs/SegmentationSpec",
            "object_detection": "#/$defs/DetectionSpec",
        },
    }
    assert {ref["$ref"] for ref in schema["oneOf"]} == {
        "#/$defs/SegmentationSpec",
        "#/$defs/DetectionSpec",
    }


def test_each_task_still_requires_its_data_and_models():
    defs = spec_schema()["$defs"]
    for name in ("SegmentationSpec", "DetectionSpec"):
        assert set(defs[name]["required"]) >= {"data", "models"}, name


def test_schema_forbids_unknown_keys():
    """extra='forbid' has to survive into the schema, or a tool built over it would
    accept the typos the engine refuses."""
    defs = spec_schema()["$defs"]
    for name in (
        "SegmentationModel",
        "DetectionModel",
        "SegmentationData",
        "DetectionData",
        "SegmentationSpec",
        "DetectionSpec",
    ):
        assert defs[name].get("additionalProperties") is False, name


def test_schema_carries_the_discriminated_unions():
    defs = spec_schema()["$defs"]
    assert "FocalLoss" in defs
    assert "TverskyLoss" in defs


def test_the_two_tasks_do_not_borrow_each_others_fields():
    """The schema is the only place the separation is visible to anything outside Python,
    so it is worth asserting there and not only in the validators' own tests."""
    defs = spec_schema()["$defs"]
    assert "colormap" in defs["SegmentationData"]["properties"]
    assert "colormap" not in defs["DetectionData"]["properties"]
    assert "classes" in defs["DetectionData"]["properties"]
    assert "classes" not in defs["SegmentationData"]["properties"]
    assert "anchors" in defs["DetectionModel"]["properties"]
    assert "anchors" not in defs["SegmentationModel"]["properties"]
    # Shared, and therefore in both rather than in neither.
    for name in ("SegmentationModel", "DetectionModel"):
        assert {"name", "input_shape", "epochs", "batch_size", "weights", "fit"} <= set(
            defs[name]["properties"]
        ), name


def test_write_schema_produces_a_file(tmp_path):
    path = write_schema(tmp_path / "spec.schema.json")
    assert path.exists()
    assert json.loads(path.read_text())["$id"].endswith("spec.schema.json")
