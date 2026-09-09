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
from pyplatypus.spec.spec import PlatypusSpec


def from_dict(config: dict[str, Any], *, source: str | None = None,
              check_paths: bool = True) -> PlatypusSpec:
    """Validate a dict into a spec, raising ConfigError with every problem at once."""
    if not isinstance(config, dict):
        raise ConfigError(
            f"a configuration must be a mapping of keys to values, got {type(config).__name__}",
            source=source,
        )
    try:
        spec = PlatypusSpec.model_validate(config)
    except ValidationError as error:
        raise ConfigError.from_validation_error(error, source=source) from None

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
