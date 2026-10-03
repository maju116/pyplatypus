"""JSON Schema export.

A machine-readable description of what a configuration may contain, generated from the
pydantic models. It is an artefact, never a second place to edit.

What it is *for* is an open question. The intention was that the R package would ship it
and catch a typo where the user made it; the R package does not, and has not needed to -
it sends the configuration to the engine and translates the engine's refusal, which keeps
one validator rather than two that can disagree. It is published for anyone writing a
configuration by hand or building a tool over one.

Since `task` arrived the top level is a choice between two shapes, so the document is a
`oneOf` and the per-task fields live under `$defs`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyplatypus.spec.spec import SPEC_ADAPTER

SCHEMA_ID = "https://maju116.github.io/platypus/schema/spec.schema.json"


def spec_schema() -> dict[str, Any]:
    schema = SPEC_ADAPTER.json_schema(mode="validation")
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
