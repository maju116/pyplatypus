"""What this package can do, asked of the package rather than of its documentation.

`available_transforms()` and `available_weights()` already answer that verb for two
things. These answer it for the rest: the losses, the metrics, the optimisers, the
callbacks, the architectures, the decodings and the tasks. The point is not convenience - it is that
the number of named components keeps growing, and **a list maintained by hand is a list
that will be wrong**. Everything below is read from the same pydantic models the schema
is generated from, so a loss added to `LossSpec` appears here with nothing else edited.

Three columns, the ones a reader needs in that order: the name you pass, the arguments
you may tune, and what it is for. The last is the component's own description - its first
paragraph, which is the summary line; the rest of it, including every measurement, is in
`spec_schema()` and in `help()` on the class.

**Detection is the case that does not fit**, and it is answered by refusing rather than by
inventing rows. A detection spec chooses no loss component (it has `box_loss`, a field on
the model, with two values) and chooses no metrics at all (mAP and matched IoU are an
output of `report`). An empty table would be a third answer and the wrong one, so both
raise with the message that says what detection actually offers.
"""

from __future__ import annotations

from functools import cache
from typing import Any, get_args

from pyplatypus.errors import PlatypusError
from pyplatypus.spec.common import Architecture, Task
from pyplatypus.spec.components import CallbackSpec, LossSpec, MetricSpec, OptimizerSpec
from pyplatypus.spec.detection import DetectionArchitecture, DetectionModel
from pyplatypus.spec.instances import DecodeSpec, InstanceModel, SeparationSpec
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.spec.schema import spec_schema

# The architectures are an enum rather than a union of models, so there is no docstring to
# read one off: `UShapedNet` is one class taking `architecture` as a switch. These notes are
# therefore written here and held by a test that every member of the enum has one - the same
# shape of promise as a description, with the gate in place of the generator.
#
# What they are allowed to say is what the code does differently and what has been measured.
# On Data Science Bowl the four span 0.9187 to 0.9217 of Dice - a spread of 0.0030 against
# 0.054 between images within any one of them - so the honest summary is that on that problem
# the architecture is not where the result comes from.
_ARCHITECTURE_NOTES = {
    Architecture.U_NET: (
        "The plain U: every skip connection concatenates the encoder's features onto the "
        "decoder's. The one to start with, and the one the measured spread above says is "
        "hard to beat on an ordinary problem."
    ),
    Architecture.U_NET_PLUS_PLUS: (
        "Nested skips: the decoder is filled in densely, so a feature map meets the encoder "
        "at several depths rather than one. It is also **the only architecture that can take "
        "`deep_supervision`** - asking for that anywhere else is refused by name, because "
        "the intermediate heads do not exist to supervise."
    ),
    Architecture.RES_U_NET: (
        "Residual convolution blocks instead of plain ones, so each block learns a "
        "correction rather than a representation. The usual argument is depth: it is the "
        "one to try when a deeper `blocks` stops helping."
    ),
    Architecture.LINKNET: (
        "Skips are **added** rather than concatenated, which halves the channels arriving in "
        "each decoder block. The cheapest of the four to run at the same width, and the "
        "reason to pick it is a memory or latency budget rather than a score."
    ),
}


def _summary(description: str) -> str:
    """A description's first paragraph, collapsed onto one line.

    The same reading the configuration reference takes of a block's description: the first
    paragraph summarises, the rest is for a reader who has chosen already. A table of six
    rows cannot carry four paragraphs each and still be a table.
    """
    paragraph = (description or "").strip().split("\n\n")[0]
    return " ".join(paragraph.split())


@cache
def _descriptions() -> dict[str, str]:
    """Every definition's own description, by definition name, read from the schema.

    From the schema rather than from `cls.__doc__` on purpose: the schema is what is
    published, and if pydantic ever stopped carrying a docstring into it, this listing
    would go empty rather than quietly disagree with the documentation site.
    """
    return {
        name: definition.get("description") or ""
        for name, definition in spec_schema()["$defs"].items()
    }


def _members(union: Any) -> list[type]:
    """The models of a discriminated union, in the order they are declared."""
    return list(get_args(get_args(union)[0]))


def _rows(union: Any) -> list[dict[str, Any]]:
    descriptions = _descriptions()
    rows = []
    for model in _members(union):
        name = model.model_fields["name"].default
        arguments = tuple(field for field in model.model_fields if field != "name")
        rows.append(
            {
                "name": name,
                "arguments": arguments,
                "description": _summary(descriptions.get(model.__name__, "")),
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def _task(task: str | Task) -> Task:
    try:
        return Task(task)
    except ValueError:
        offered = ", ".join(sorted(member.value for member in Task))
        raise PlatypusError(f"unknown task {task!r}. This package does: {offered}") from None


def available_tasks() -> list[dict[str, Any]]:
    """What this package can be asked to do at all.

    Three rows, and the weakest of this family as a list - it earns its place as the entry
    point, which is the question somebody asks before they have read anything.

    Returns:
        One row per task: the value you write under `task`, the components it lets you
        choose, and what it is for.

    >>> [row["name"] for row in available_tasks()]
    ['instance_segmentation', 'object_detection', 'semantic_segmentation']
    >>> next(r for r in available_tasks() if r["name"] == "instance_segmentation")["chooses"]
    ('models', 'separation', 'decode', 'loss', 'metrics', 'optimizer', 'callbacks', 'augmentation')
    """
    descriptions = _descriptions()
    rows = [
        {
            "name": Task.SEMANTIC_SEGMENTATION.value,
            "chooses": ("models", "loss", "metrics", "optimizer", "callbacks", "augmentation"),
            "description": _summary(descriptions.get("SegmentationSpec", ""))
            or "A mask per class, scored by overlap.",
        },
        {
            "name": Task.INSTANCE_SEGMENTATION.value,
            # `separation` and `decode` are listed beside the components every task
            # chooses, because they are the two the task exists for and a reader scanning
            # this table should not have to open the model spec to find out it has them.
            "chooses": (
                "models",
                "separation",
                "decode",
                "loss",
                "metrics",
                "optimizer",
                "callbacks",
                "augmentation",
            ),
            "description": _summary(descriptions.get("InstanceSpec", ""))
            or "Which object every pixel belongs to, scored as instances.",
        },
        {
            "name": Task.OBJECT_DETECTION.value,
            "chooses": ("models", "optimizer", "callbacks", "augmentation"),
            "description": _summary(descriptions.get("DetectionSpec", ""))
            or "Boxes with labels, scored by average precision.",
        },
    ]
    return sorted(rows, key=lambda row: row["name"])


def available_models(task: str | Task = Task.SEMANTIC_SEGMENTATION) -> list[dict[str, Any]]:
    """The architectures this package can build for a task.

    Returns:
        One row per architecture: the name you write under `architecture`, the arguments the
        model block accepts, and what the architecture does differently.

    >>> sorted(row["name"] for row in available_models())
    ['linknet', 'res_u_net', 'u_net', 'u_net_plus_plus']
    >>> [row["name"] for row in available_models("object_detection")]
    ['yolo3']
    >>> "separation" in available_models("instance_segmentation")[0]["arguments"]
    True
    """
    if _task(task) is Task.INSTANCE_SEGMENTATION:
        # The same four architectures - an instance model is a U-shaped network with a
        # wider head, which is the argument `#160` settled the shape on: folding CellPose
        # and HoVer-Net into `architecture` would make "CellPose on a LinkNet backbone"
        # inexpressible. What differs is the arguments, so those are read from the
        # instance model rather than from the semantic one.
        arguments = tuple(
            field for field in InstanceModel.model_fields if field not in ("name", "architecture")
        )
        return [
            {
                "name": member.value,
                "arguments": arguments,
                "description": _ARCHITECTURE_NOTES[member],
            }
            for member in Architecture
        ]
    if _task(task) is Task.OBJECT_DETECTION:
        arguments = tuple(
            field for field in DetectionModel.model_fields if field not in ("name", "architecture")
        )
        return [
            {
                "name": member.value,
                "arguments": arguments,
                "description": _summary(_descriptions().get("DetectionModel", "")),
            }
            for member in DetectionArchitecture
        ]

    arguments = tuple(
        field for field in SegmentationModel.model_fields if field not in ("name", "architecture")
    )
    return [
        {"name": member.value, "arguments": arguments, "description": _ARCHITECTURE_NOTES[member]}
        for member in Architecture
    ]


def available_losses(task: str | Task = Task.SEMANTIC_SEGMENTATION) -> list[dict[str, Any]]:
    """The losses you can name under `loss`.

    Returns:
        One row per loss: the name, the arguments it accepts, and what it is for.

    Raises:
        PlatypusError: for `object_detection`, which chooses no loss component. Its
            objective is fixed and `box_loss` on the model picks the box term.

    >>> [row["name"] for row in available_losses()][:3]
    ['boundary', 'cce', 'cce_dice']
    >>> next(row for row in available_losses() if row["name"] == "tversky")["arguments"]
    ('alpha', 'smooth')
    """
    if _task(task) is Task.OBJECT_DETECTION:
        box_loss = spec_schema()["$defs"]["DetectionModel"]["properties"]["box_loss"]
        offered = ", ".join(str(value) for value in box_loss.get("enum", ()))
        raise PlatypusError(
            "object_detection chooses no loss component: the detector's objective is fixed "
            f"(objectness, class and box terms) and the only choice is `box_loss` on the "
            f"model, which takes {offered}. On BCCD the two are inside the noise on average "
            "precision and `offsets` is 60 times more reproducible on matched overlap, which "
            "is why it is the default."
        )
    return _rows(LossSpec)


def available_metrics(task: str | Task = Task.SEMANTIC_SEGMENTATION) -> list[dict[str, Any]]:
    """The metrics you can name under `metrics`.

    Returns:
        One row per metric: the name, the arguments it accepts, and what it measures.

    Raises:
        PlatypusError: for `object_detection`, which chooses no metrics at all - they are
            an output of `report` rather than a setting.

    >>> [row["name"] for row in available_metrics()]
    ['cldice', 'dice', 'iou', 'tversky']
    """
    if _task(task) is Task.OBJECT_DETECTION:
        raise PlatypusError(
            "object_detection chooses no metrics: every detection run reports the same set, "
            "`map_50`, `map_50_95` and `mean_matched_iou`, with average precision per class "
            "beside them. They come out of `report` rather than going into the spec, which "
            "is why there is nothing here to list."
        )
    return _rows(MetricSpec)


def available_separations() -> list[dict[str, Any]]:
    """The separation signals you can name under `separation`, and what each one costs.

    `instance_segmentation` only, and one block per model rather than a list: this is what
    the network is trained to predict, so it sets the width of the head and which losses
    are admissible, and several targets do not share a training run.

    Returns:
        One row per signal: the name, the arguments its block accepts, how wide a head it
        needs at `n_class` 2, and what it predicts. The widths differ by where the extra
        channels go - `rim` adds a third class, `flows` adds two continuous channels in
        front of the classes - which is why nothing states a head width in a file.

    >>> [row["name"] for row in available_separations()]
    ['flows', 'rim']
    >>> next(r for r in available_separations() if r["name"] == "rim")["arguments"]
    ('width',)
    >>> next(r for r in available_separations() if r["name"] == "flows")["arguments"]
    ('mode', 'loss', 'weight')
    >>> [(r["name"], r["head_channels"]) for r in available_separations()]
    [('flows', 4), ('rim', 3)]
    """
    rows = _rows(SeparationSpec)
    for row, model in zip(
        sorted(rows, key=lambda row: row["name"]),
        sorted(_members(SeparationSpec), key=lambda m: m.model_fields["name"].default),
        strict=True,
    ):
        row["head_channels"] = model.signal.head_channels(2)
    return rows


def available_decodings() -> list[dict[str, Any]]:
    """The decodings you can name in `decode`, and which separation signal each one reads.

    `instance_segmentation` only - the other two tasks have nothing to decode - and the
    one row of this family that carries a constraint rather than only a description: the
    two axes are a tree, so a decoding says which targets it can be fed.

    Returns:
        One row per decoding: the name, the arguments it accepts, which `separation`
        values it reads, and what it does. An empty `needs` means every target, which is
        `connected_components` and is why that one is the yardstick.

    >>> [row["name"] for row in available_decodings()]
    ['connected_components', 'follow_flows', 'nearest_seed', 'watershed']
    >>> next(r for r in available_decodings() if r["name"] == "follow_flows")["needs"]
    ('flows',)
    >>> next(r for r in available_decodings() if r["name"] == "connected_components")["needs"]
    ()
    """
    rows = _rows(DecodeSpec)
    for row, model in zip(
        sorted(rows, key=lambda row: row["name"]),
        sorted(_members(DecodeSpec), key=lambda m: m.model_fields["name"].default),
        strict=True,
    ):
        row["needs"] = tuple(sorted(signal.value for signal in model.needs))
    return rows


def available_optimizers() -> list[dict[str, Any]]:
    """The optimisers you can name under `optimizer`. The same set for either task.

    Returns:
        One row per optimiser: the name, the arguments it accepts, and what it is for.

    >>> [row["name"] for row in available_optimizers()][:3]
    ['adadelta', 'adagrad', 'adam']
    >>> len(available_optimizers())
    8
    """
    return _rows(OptimizerSpec)


def available_callbacks() -> list[dict[str, Any]]:
    """The callbacks you can name under `callbacks`. The same set for either task.

    Returns:
        One row per callback: the name, the arguments it accepts, and what it is for.

    >>> next(row for row in available_callbacks() if row["name"] == "early_stopping")["arguments"]
    ('monitor', 'patience', 'min_delta', 'restore_best')
    """
    return _rows(CallbackSpec)
