"""JSON Schema export.

This is what lets the R package catch a typo where the user made it, instead of the
error arriving from Python after the fact. The schema is generated from the pydantic
models at build time and shipped inside the R package; it is an artefact, never a second
place to edit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyplatypus.spec.spec import PlatypusSpec

SCHEMA_ID = "https://maju116.github.io/platypus/schema/spec.schema.json"


def spec_schema() -> dict[str, Any]:
    schema = PlatypusSpec.model_json_schema(mode="validation")
    schema["$id"] = SCHEMA_ID
    schema["title"] = "platypus experiment specification"
    return schema


def write_schema(path: str | Path = "schema/spec.schema.json") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec_schema(), indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:  # console script: platypus-schema
    print(write_schema())
