"""The JSON Schema is shipped to R, so it has to be real and it has to be stable."""

import json

from pyplatypus import spec_schema, write_schema


def test_schema_is_json_serialisable():
    json.dumps(spec_schema())


def test_schema_describes_the_top_level():
    schema = spec_schema()
    assert schema["$id"].endswith("spec.schema.json")
    assert set(schema["required"]) >= {"data", "models"}


def test_schema_forbids_unknown_keys():
    """extra='forbid' has to survive into the schema or R will not catch typos."""
    schema = spec_schema()
    model = schema["$defs"]["SegmentationModel"]
    assert model.get("additionalProperties") is False


def test_schema_carries_the_discriminated_unions():
    defs = spec_schema()["$defs"]
    assert "FocalLoss" in defs
    assert "TverskyLoss" in defs


def test_write_schema_produces_a_file(tmp_path):
    path = write_schema(tmp_path / "spec.schema.json")
    assert path.exists()
    assert json.loads(path.read_text())["$id"].endswith("spec.schema.json")
