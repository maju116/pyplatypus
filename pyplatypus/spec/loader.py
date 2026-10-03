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

    A configuration with no `task` is a segmentation one, because every configuration
    written before detection existed is. The default is applied here rather than on the
    field: a discriminated union needs its tag present in the input to choose a branch at
    all, so there is nowhere else to put it. On a copy, since a caller's dict is theirs.
    """
    if not isinstance(config, dict):
        raise ConfigError(
            f"a configuration must be a mapping of keys to values, got {type(config).__name__}",
            source=source,
        )
    if "task" not in config:
        config = {**config, "task": Task.SEGMENTATION.value}
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
