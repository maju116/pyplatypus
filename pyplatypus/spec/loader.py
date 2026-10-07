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


def from_dict(
    config: dict[str, Any], *, source: str | None = None, check_paths: bool = True
) -> PlatypusSpec:
    """Validate a dict into a spec, raising ConfigError with every problem at once.

    **`task` is required.** It was briefly defaulted to segmentation so that files written
    before detection existed kept working, and that default is gone: a configuration that
    does not say what it is asking for means whatever the version reading it decides, and
    more tasks are coming.

    The default also cost the wrong person. A *detection* configuration with no `task` was
    validated as segmentation and reported `classes` and `anchors_per_grid` as **extra
    fields**, never naming the tag - precisely the confusion a discriminator exists to
    prevent.

    Args:
        config: The configuration, with `task`, `data` and `models` at the top level. The
            keys are the ones written in the YAML, because this is what reads the YAML.
        source: A name for the configuration in error messages - a file path, usually.
            `from_yaml` passes the path; pass something when the dict came from somewhere
            a reader could go and look.
        check_paths: Whether the paths in `data` must exist. True wherever a run is about
            to happen, so a typo is caught before anything is read; False to validate the
            shape of a configuration on a machine that does not hold the data - a test, a
            schema check, or a specification being written.

    Returns:
        A `SegmentationSpec` or a `DetectionSpec`, whichever `task` names. Both are
        `PlatypusSpec`, so anything needing only a name, a seed or a rank takes either.

    Raises:
        ConfigError: With **every** problem listed rather than the first. A configuration
                with four mistakes should take one run to fix, not four.

    >>> spec = from_dict(
    ...     {
    ...         "task": "semantic_segmentation",
    ...         "data": {
    ...             "train_path": "train/",
    ...             "validation_path": "valid/",
    ...             "colormap": [[0, 0, 0], [255, 255, 255]],
    ...         },
    ...         "models": [
    ...             {"name": "unet", "architecture": "u_net", "input_shape": [256, 256]}
    ...         ],
    ...     },
    ...     check_paths=False,
    ... )
    >>> spec.task.value, spec.rank, len(spec.models)
    ('semantic_segmentation', 2, 1)

    Defaults come from the engine, so a short configuration is a complete one:

    >>> spec.models[0].blocks, spec.models[0].filters, spec.models[0].loss.name
    (4, 16, 'cce')

    A missing `task` is refused by name rather than guessed at:

    >>> from_dict({"data": {}, "models": []}, check_paths=False)
    Traceback (most recent call last):
    pyplatypus.errors.ConfigError: ...
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
            error,
            source=source,
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
    """Read a YAML file into a spec: the same object `from_dict` builds, by the same rules.

    One pipeline, two ways in. A specification written by hand and one read from a file are
    the same thing, and `as_dict()` turns it back - which is how a worked example can write
    out the YAML of the run it has just done, beside the results.

    Args:
        path: The file to read.
        check_paths: As for `from_dict`.

    Returns:
        A `SegmentationSpec` or a `DetectionSpec`, whichever `task` names.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, is empty, or describes a
                configuration with problems. Each is said with the path, so the reader knows
                which file to open.

    >>> import pathlib, tempfile
    >>> lines = [
    ...     'task: semantic_segmentation',\n    ...     'data:',\n    ...     '  train_path: train/',\n    ...     '  validation_path: valid/',\n    ...     '  colormap: [[0, 0, 0], [255, 255, 255]]',\n    ...     'models:',\n    ...     '  - name: unet',\n    ...     '    architecture: u_net',\n    ...     '    input_shape: [256, 256]',
    ... ]
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "run.yaml"
    >>> _ = path.write_text("\\n".join(lines))
    >>> spec = from_yaml(path, check_paths=False)
    >>> spec.models[0].name, spec.models[0].input_shape
    ('unet', (256, 256))
    """
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
