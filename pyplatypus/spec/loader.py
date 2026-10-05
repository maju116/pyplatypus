"""Building a spec from YAML or from a plain dict.

R hands us a dict (reticulate marshals an R list into one), YAML gives us a dict too, so
both routes land in `from_dict` and there is only ever one code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pyplatypus.errors import ConfigError
from pyplatypus.spec.common import Task
from pyplatypus.spec.spec import SPEC_ADAPTER, PlatypusSpec


def from_dict(config: dict[str, Any], *, source: str | None = None,
              check_paths: bool = True) -> PlatypusSpec:
    """Validate a dict into a spec, raising ConfigError with every problem at once.

    **`task` is required.** It was briefly defaulted to segmentation so that files written
    before detection existed kept working, and that default is gone: a configuration that
    does not say what it is asking for means whatever the version reading it decides, and
    more tasks are coming.

    The default also cost the wrong person. A *detection* configuration with no `task` was
    validated as segmentation and reported `classes` and `anchors_per_grid` as **extra
    fields**, never naming the tag - precisely the confusion a discriminator exists to
    prevent.
    """
    if not isinstance(config, dict):
        raise ConfigError(
            f"a configuration must be a mapping of keys to values, got {type(config).__name__}",
            source=source,
        )
    _check_task(config, source=source)
    try:
        spec = SPEC_ADAPTER.validate_python(config)
    except ValidationError as error:
        # The union's tag leads every nested location - 'detection.models[0].anchors' -
        # and reads as a field called 'detection'. Only the caller knows the tag.
        tag = config.get("task")
        raise ConfigError.from_validation_error(
            error, source=source,
            drop_prefix=tag if isinstance(tag, str) else None,
        ) from None

    if check_paths:
        problems = spec.check_paths()
        if problems:
            raise ConfigError(
                "The configuration points at data that is not there:\n"
                + "\n".join(f"  - {p}" for p in problems),
                problems=[{"where": "data", "problem": p} for p in problems],
                source=source,
            )
    return spec


def from_yaml(path: str | Path, *, check_paths: bool = True) -> PlatypusSpec:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"configuration file '{path}' does not exist")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ConfigError(f"'{path}' is not valid YAML:\n  {error}", source=str(path)) from None
    if raw is None:
        raise ConfigError(f"'{path}' is empty", source=str(path))
    return from_dict(raw, source=str(path), check_paths=check_paths)


#: What these two were called before they were spelled out, and the only values an existing
#: file or an older script is likely to hold. Named back on sight, rather than left to
#: pydantic's "input should be ..." list: a rename the reader has to deduce from a list of
#: valid values is a rename done to them rather than for them.
_RENAMED = {
    "segmentation": Task.SEMANTIC_SEGMENTATION.value,
    "detection": Task.OBJECT_DETECTION.value,
}


def _check_task(config: dict[str, Any], *, source: str | None = None) -> None:
    """Refuse a missing or outdated `task` before the union gets a chance to guess."""
    tasks = ", ".join(repr(task.value) for task in Task)
    if "task" not in config:
        raise ConfigError(
            f"a configuration has to say what it is asking for: add `task`, one of "
            f"{tasks}. It decides the shape of `data` and of every entry in `models`, so "
            f"there is nothing safe to assume.",
            source=source,
        )
    given = config["task"]
    if isinstance(given, str) and given in _RENAMED:
        raise ConfigError(
            f"`task: {given}` was renamed to `{_RENAMED[given]}`. Both tasks are spelled "
            f"out now, because 'segmentation' stops naming one thing as soon as instance "
            f"segmentation exists.",
            source=source,
        )
