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
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def wants_a_record(spec) -> bool:
    """Whether `output_dir` was asked for rather than defaulted into existence."""
    return "output_dir" in spec.model_fields_set


def _history_parts(history: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The per-epoch rows, and the run-level facts that sit beside them.

    `History.to_dict()` is the one place that says what a history carries, and until
    `pyplatypus#417` it had **no caller at all**. So `weights_epoch` and `stop_reason` were
    recorded where they are decided, read by all four score tables, and then left the process
    with the session: a directory six months later said which epoch was best by every metric
    the run logged and not which epoch's weights it shipped, in the one file whose docstring
    calls it a thing you can run rather than read.

    Reading them off `to_dict()` rather than naming them here is what keeps this from being a
    second list to maintain - a field added to `History` reaches the record by arriving, which
    is the shape §3F prefers over two places agreeing.

    `history` can also be a bare list of rows: `fit` passes on whatever it was given, and the
    fallback is what this module has always done with it.
    """
    if not hasattr(history, "to_dict"):
        return list(history or []), {}
    carried = dict(history.to_dict())
    rows = carried.pop("records", None) or []
    return rows, carried


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
        history: The per-epoch numbers, as the trainer produced them. Whatever run-level
            facts it carries beside those rows - `weights_epoch`, `stop_reason` - are
            written as siblings of `history` rather than inside it, read off
            `History.to_dict()` so that a field added there needs no edit here.

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

    rows, run_level = _history_parts(history)
    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "pyplatypus": __version__,
        "model": name,
        # The whole thing, so the record is a thing you can run rather than read. A
        # detector's fitted anchors are *not* in here - they are under `derived`, because
        # the specification that produced them did not name them.
        "specification": spec.to_dict(),
        "derived": derived or {},
        "history": rows,
    }
    # The run-level facts go BESIDE the rows rather than inside them, because `history` is a
    # list of per-epoch rows and R reads exactly that key - `read_record()` turns it into a
    # data frame. A scalar in there would change the key's type for the one reader that
    # exists, which is the whole reason this is additive.
    #
    # A clash is a refusal rather than a silent replacement: a `model` field added to
    # `History` would otherwise overwrite the model's name, and a record quietly saying the
    # wrong thing is worse than one that could not be written. A plain `ValueError` and not a
    # `PlatypusError`, because this cannot be reached by anything a user wrote - it is this
    # module's two dicts disagreeing, and R documents the `PlatypusError` kinds as conditions
    # a caller can meet.
    clash = sorted(set(run_level) & set(record))
    if clash:
        raise ValueError(
            f"a history carries {clash}, which the record already uses for something else. "
            "Rename the field on `History` or give it its own key here - writing it would "
            "overwrite what the record says about the run"
        )
    record.update(run_level)
    path = directory / "run.json"
    # `allow_nan=False` after `_without_nonfinite` has already replaced every NaN and
    # infinity with `null`: the walk handles the case a run really produces, and the flag
    # turns any route it missed into a loud failure here rather than a file that cannot be
    # read later. `json.dumps` otherwise writes the bare token `NaN`, which Python reads
    # back and `jsonlite` refuses outright - so the record of the one run somebody wants
    # to inspect, the one that went to NaN, was the one R could not open.
    # `pyplatypus#373`.
    text = json.dumps(_without_nonfinite(record), indent=2, default=_plain, allow_nan=False)
    path.write_text(text + "\n")
    return path


def read_record(path: str | Path) -> dict[str, Any]:
    """A record back off disk. Plain data; `from_dict(record["specification"])` runs it.

    Args:
        path: The `run.json` to read.

    Returns:
        The record as plain data: `specification`, `pyplatypus`, `history`, `derived` if
        the run had any, and the run-level facts beside the history - `weights_epoch`, the
        epoch the surviving weights came from or `None` when they are not any one epoch's,
        and `stop_reason`. Nothing is reconstructed into objects, because the record is a
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


def _without_nonfinite(value):
    """The record with every NaN and infinity replaced by `None`, which JSON writes as `null`.

    A history legitimately holds a NaN: `dice` with `smooth: 0` on a class no image in the
    batch contains is 0/0, and that is what `terminate_on_nan` watches for. JSON has no
    literal for it, so `json.dumps` writes the bare token `NaN` by extension. Python reads
    that back, which is why nothing here noticed; `jsonlite::fromJSON` refuses it with a
    lexical error, so `read_record()` on the R side could not open the record at all.

    `null` reads as `NA` in R and as `None` in Python, which is what a number that is not
    a number should look like to a reader. numpy scalars are unwrapped on the way, because
    a `float32` NaN is not an instance of `float`.
    """
    if isinstance(value, dict):
        return {key: _without_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_without_nonfinite(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (AttributeError, ValueError):  # pragma: no cover - not a scalar
            return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
