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
    """The JSON Schema for a specification, generated from the models themselves.

    Not hand-written, so it cannot describe a configuration format the engine does not
    have. This is also what the YAML reference on both documentation sites is built from,
    which is why every field carries a `description` and a test asserts it.

    Returns:
        The schema as plain data, with `$defs` holding one definition per model and
        `oneOf` at the top selecting on `task`.

    >>> schema = spec_schema()
    >>> schema["title"]
    'platypus experiment specification'
    >>> sorted(schema["$defs"])[:3]
    ['Activation', 'Adadelta', 'Adagrad']
    >>> "SegmentationModel" in schema["$defs"], "DetectionModel" in schema["$defs"]
    (True, True)
    >>> schema["$defs"]["SplitSpec"]["properties"]["group_by"]["description"][:24]
    '**Required, and may be n'
    """
    schema = SPEC_ADAPTER.json_schema(mode="validation")
    schema["$id"] = SCHEMA_ID
    schema["title"] = "platypus experiment specification"
    return schema


def write_schema(path: str | Path = "schema/spec.schema.json") -> Path:
    """Write the schema to a file, creating the directory if it is not there.

    Also the `platypus-schema` console script. An editor that understands JSON Schema will
    complete and check a YAML configuration against it, which is the cheapest way to find
    a misspelled key - before a run rather than during one.

    Args:
        path: Where to write. The default is the path the repository keeps it at.

    Returns:
        The path written, so a caller can print it.

    >>> import json, pathlib, tempfile
    >>> written = write_schema(pathlib.Path(tempfile.mkdtemp()) / "spec.schema.json")
    >>> written.name
    'spec.schema.json'
    >>> json.loads(written.read_text())["title"]
    'platypus experiment specification'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec_schema(), indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:  # console script: platypus-schema
    print(write_schema())
