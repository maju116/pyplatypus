"""What a run leaves behind, beyond the weights.

The weights are not the whole of a trained model. A segmentation model is meaningless
without the colormap that says what its channels are; a detector is meaningless without the
anchors its boxes are relative to - and when those were fitted rather than named, the run is
the only place they exist. Both were recoverable only by remembering to export weights, and
anything that depends on remembering is a thing that gets lost.

So `fit()` writes a record. **Only when `output_dir` was given**, which pydantic can answer
honestly through `model_fields_set`: a field with a default cannot otherwise be told apart
from one the user set to that default, and writing files into somebody's working directory
because a default exists is not a thing to do by surprise.

The record is deliberately the *specification* plus what the run derived from it, rather
than a summary. A summary is for reading; a specification is for running again.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def wants_a_record(spec) -> bool:
    """Whether `output_dir` was asked for rather than defaulted into existence."""
    return "output_dir" in spec.model_fields_set


def write_record(
    spec, name: str, *, derived: dict[str, Any] | None = None, history: Any = None
) -> Path:
    """Write `<output_dir>/<name>/run.json` and return its path.

    `derived` is whatever the run worked out that the specification does not already say -
    fitted anchors, the target survey. For a segmentation run it is usually empty, and that
    is the honest answer rather than an omission: the specification already carries the
    colormap, the input shape and the window, so re-running it reproduces the model.

    Args:
        spec: The specification the run was built from. It is written out whole, so the
            record is enough to reproduce the run without the file it came from.
        name: The model's name, which is also the directory the record goes in.
        derived: What the run worked out for itself, if anything.
        history: The per-epoch numbers, as the trainer produced them.

    Returns:
        The path written: `<output_dir>/<name>/run.json`.

    The record stamps `pyplatypus.__version__`, which is why that literal and the one in
    `pyproject.toml` are held together by a test: a release reporting the wrong version
    writes wrong provenance into every file it produces, and those files outlive it.

    No doctest: this writes a directory and wants a whole specification to do it, and the
    engine's own tests cover it with one.
    """
    from pyplatypus import __version__

    directory = Path(spec.output_dir) / name
    directory.mkdir(parents=True, exist_ok=True)

    record = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pyplatypus": __version__,
        "model": name,
        # The whole thing, so the record is a thing you can run rather than read. A
        # detector's fitted anchors are *not* in here - they are under `derived`, because
        # the specification that produced them did not name them.
        "specification": spec.to_dict(),
        "derived": derived or {},
        "history": getattr(history, "records", history) or [],
    }
    path = directory / "run.json"
    path.write_text(json.dumps(record, indent=2, default=_plain) + "\n")
    return path


def read_record(path: str | Path) -> dict[str, Any]:
    """A record back off disk. Plain data; `from_dict(record["specification"])` runs it.

    Args:
        path: The `run.json` to read.

    Returns:
        The record as plain data: `specification`, `pyplatypus`, `history`, and `derived`
        if the run had any. Nothing is reconstructed into objects, because the record is a
        description of what happened and not a live thing.

    >>> import json, pathlib, tempfile
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "run.json"
    >>> _ = path.write_text(json.dumps({"pyplatypus": "0.7.0a1", "history": []}))
    >>> read_record(path)["pyplatypus"]
    '0.7.0a1'
    """
    return json.loads(Path(path).read_text())


def _plain(value):
    """numpy scalars and tuples, which json does not know and a run produces plenty of."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)
