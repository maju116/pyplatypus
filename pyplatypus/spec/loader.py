"""Building a spec from YAML or from a plain dict.

R hands us a dict (reticulate marshals an R list into one), YAML gives us a dict too, so
both routes land in `from_dict` and there is only ever one code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import yaml
from pydantic import BaseModel, ValidationError

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
    the same thing, and `save_spec()` turns it back - which is how a worked example can write
    out the YAML of the run it has just done, beside the results. This sentence named
    `as_dict()` until 2026-10-11, a method no spec has ever had; the way back did not exist
    at all, which is what platypus#120 was filed about.

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


def save_spec(spec: PlatypusSpec, path: str | Path, *, defaults: bool = False) -> Path:
    """Write a specification back out as YAML - what was set, not what it holds.

    `from_yaml` reads a configuration and `PlatypusSpec` builds one in code; this is the way
    back, so the two interfaces are one pipeline rather than one and a half. A worked example
    can write the YAML of the run it has just done beside the results, and a specification
    assembled in R or Python can become the file somebody else runs.

    **It writes the fields that were set.** Not the fields the object holds, which on a
    two-field model is 29 of them - and the difference is not verbosity. A model given
    `weights` and no `architecture` adopts the architecture from the weights sidecar, and that
    adoption fires on `architecture` *not being in* `model_fields_set`. Dump the defaults into
    the file and the reloaded specification has all 29 set, so the adoption can never fire
    again: the file would look like the original and quietly stop doing what it did. R paid
    for this once already in the other direction (`JOURNAL.md` §4aa) and dropped its own
    defaults for the same reason.

    `defaults=True` is the opposite want, which is legitimate and is a second mode rather than
    the default: freeze this run exactly as it resolved, so it reproduces in five years. The
    cost is the one above - the file stops adopting anything, by design.

    **Every discriminator is always written**, whatever was set. `task` is the obvious one -
    it is what `from_yaml` dispatches on, and a configuration that does not say what it is
    asking for means whatever the version reading it decides. The same is true one level
    down and is easier to miss: `DiceLoss(smooth=0.5)` sets `smooth` and not `name`, because
    `name` is a `Literal` with a default, so a dump of what was set alone writes
    `loss: {smooth: 0.5}` - a block with no tag, which comes back as *Unable to extract tag
    using discriminator 'name'*. A field annotated `Literal` is a tag here, and tags survive.

    Args:
        spec: The specification to write. Either task.
        path: Where to write it. The parent directory must exist.
        defaults: Write every field at its resolved value, including the ones nobody set.

    Returns:
        The path written, so a caller can log or return it.

    >>> import tempfile, pathlib
    >>> from pyplatypus.spec import SegmentationSpec, SegmentationData, SegmentationModel
    >>> spec = SegmentationSpec(
    ...     data=SegmentationData(
    ...         train_path="train/", validation_path="valid/",
    ...         colormap=[[0, 0, 0], [255, 255, 255]],
    ...     ),
    ...     models=[SegmentationModel(name="m", input_shape=(32, 32))],
    ... )
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "spec.yaml"
    >>> _ = save_spec(spec, path)
    >>> sorted(yaml.safe_load(path.read_text())["models"][0])
    ['input_shape', 'name']
    >>> frozen = save_spec(spec, path.with_name("frozen.yaml"), defaults=True)
    >>> len(yaml.safe_load(frozen.read_text())["models"][0])
    29
    """
    path = Path(path)
    full = spec.model_dump(mode="json")
    data = (
        full
        if defaults
        else _with_tags(spec, spec.model_dump(mode="json", exclude_unset=True), full)
    )
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _is_tag(annotation: Any) -> bool:
    """Whether a field is a discriminator, which is to say a `Literal` of exactly one value.

    "Annotated `Literal`" was the first rule here and it was too wide: `box_loss` is
    `Literal["offsets", "giou"]` and `annotation_format` is `Literal["pascal_voc",
    "labelme"]`, both of them choices a user makes, and writing them unset would freeze a
    default into the file - the very thing this function exists to avoid. A discriminator
    pins one value per member, so counting the values separates the two exactly: measured
    across the whole spec, 31 single-value Literal fields, all of them `name` or `task`, and
    the only two with more are those choices.
    """
    return get_origin(annotation) is Literal and len(get_args(annotation)) == 1


def _with_tags(model: BaseModel, lean: dict[str, Any], full: dict[str, Any]) -> dict[str, Any]:
    """Put every discriminator back into a dump of only what was set.

    pydantic's own serialisation does the converting - tuples to lists, enums to their
    values - so this walks the tree beside its two outputs and copies back the one kind of
    field that `exclude_unset` is wrong about. An unset block stays absent: a specification
    that never named a loss should come back with the default loss, not with a written one.
    """
    # `type(model)` rather than the instance: pydantic 2.11 deprecated reading `model_fields`
    # off an instance, and a deprecation warning per field is 81 of them on one small spec.
    fields = type(model).model_fields
    for field, info in fields.items():
        if _is_tag(info.annotation) and field in full:
            lean[field] = full[field]
    for field in fields:
        if field not in lean:
            continue
        value = getattr(model, field)
        if isinstance(value, BaseModel):
            _with_tags(value, lean[field], full[field])
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, BaseModel):
                    _with_tags(item, lean[field][index], full[field][index])
    # Declaration order, not insertion order: a tag put back lands at the end of the dict,
    # and `task` belongs at the top of the file where a reader looks for it.
    return {field: lean[field] for field in fields if field in lean}
