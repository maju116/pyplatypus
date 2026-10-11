"""`save_spec` writes what was set, and the round trip is what proves it.

The naive version of this function is one line - `yaml.safe_dump(spec.to_dict())` - and it
produces a file that looks like the original and quietly stops doing what the original did.
A model given `weights` and no `architecture` adopts the architecture from the weights
sidecar, and the adoption fires on `architecture` **not being in** `model_fields_set`
(`engine.py`, `detection_engine.py`). Dumping defaults sets all 29 fields, so it can never
fire again.

So the test that matters is not "the file parses". It is that `model_fields_set` survives,
at every level of the tree - and a parse test passes straight through the bug.
"""

from pathlib import Path

import pytest
import yaml

from pyplatypus.spec import (
    DetectionData,
    DetectionModel,
    DetectionSpec,
    SegmentationData,
    SegmentationModel,
    SegmentationSpec,
    from_yaml,
    save_spec,
)
from pyplatypus.spec.components import DiceLoss, EarlyStopping, Sgd


def minimal() -> SegmentationSpec:
    return SegmentationSpec(
        data=SegmentationData(
            train_path="train/",
            validation_path="valid/",
            colormap=[[0, 0, 0], [255, 255, 255]],
        ),
        models=[SegmentationModel(name="m", input_shape=(32, 32))],
    )


def rich() -> SegmentationSpec:
    return SegmentationSpec(
        seed=7,
        data=SegmentationData(
            train_path="train/",
            validation_path="valid/",
            colormap=[[0, 0, 0], [255, 255, 255]],
        ),
        models=[
            SegmentationModel(
                name="m",
                input_shape=(32, 32),
                loss=DiceLoss(smooth=0.5),
                optimizer=Sgd(momentum=0.9, nesterov=True),
                callbacks=[EarlyStopping(patience=3)],
            )
        ],
    )


def fields_set(spec) -> dict[str, set[str]]:
    """`model_fields_set` for the spec and for every model underneath it.

    Flattened rather than compared object by object, because the failure this guards against
    is a field appearing somewhere in the tree, and a shallow comparison on the top level
    would miss every one of them: the top level of `minimal()` has `data` and `models` set
    either way.
    """
    found = {"": set(spec.model_fields_set), "data": set(spec.data.model_fields_set)}
    for index, model in enumerate(spec.models):
        found[f"models[{index}]"] = set(model.model_fields_set)
        for field in ("loss", "optimizer"):
            component = getattr(model, field, None)
            if component is not None:
                found[f"models[{index}].{field}"] = set(component.model_fields_set)
        for position, callback in enumerate(model.callbacks or ()):
            found[f"models[{index}].callbacks[{position}]"] = set(callback.model_fields_set)
    return found


@pytest.mark.parametrize("build", [minimal, rich], ids=["minimal", "rich"])
def test_model_fields_set_survives_the_round_trip(build, tmp_path: Path):
    spec = build()
    reloaded = from_yaml(save_spec(spec, tmp_path / "spec.yaml"), check_paths=False)

    before, after = fields_set(spec), fields_set(reloaded)
    # The discriminators are written whatever was set, because a block with no tag cannot be
    # read back at all, so they are the fields allowed to appear - and nothing else is. Both
    # directions: a field that was set and is now absent is the rot this guards.
    assert set(before) == set(after)
    for where in before:
        assert after[where] - before[where] <= {"task", "name"}, where
        assert before[where] - after[where] == set(), where


def test_the_adoption_a_naive_dump_would_break(tmp_path: Path):
    """The concrete case from platypus#120, asserted on the mechanism rather than on a run:
    `architecture` unset is what lets the weights sidecar decide it."""
    spec = SegmentationSpec(
        data=SegmentationData(
            train_path="train/",
            validation_path="valid/",
            colormap=[[0, 0, 0], [255, 255, 255]],
        ),
        models=[SegmentationModel(name="m", input_shape=(32, 32), weights="dsbowl-unet")],
    )
    assert "architecture" not in spec.models[0].model_fields_set

    kept = from_yaml(save_spec(spec, tmp_path / "kept.yaml"), check_paths=False)
    assert "architecture" not in kept.models[0].model_fields_set

    frozen = from_yaml(save_spec(spec, tmp_path / "frozen.yaml", defaults=True), check_paths=False)
    assert "architecture" in frozen.models[0].model_fields_set


def test_what_the_two_modes_write(tmp_path: Path):
    """The measurement in platypus#120 and in the docstring: two fields against 29."""
    spec = minimal()
    lean = yaml.safe_load(save_spec(spec, tmp_path / "lean.yaml").read_text())
    frozen = yaml.safe_load(save_spec(spec, tmp_path / "frozen.yaml", defaults=True).read_text())
    assert sorted(lean["models"][0]) == ["input_shape", "name"]
    assert len(frozen["models"][0]) == 29
    assert lean["task"] == frozen["task"] == "semantic_segmentation"


def test_a_detection_spec_goes_round_too(tmp_path: Path):
    """Either task, and the tag is the thing that makes the file readable at all."""
    spec = DetectionSpec(
        data=DetectionData(
            train_path="train/",
            validation_path="valid/",
            classes=["rbc", "wbc"],
            annotation_format="pascal_voc",
        ),
        task="object_detection",
        models=[DetectionModel(name="d", input_shape=(416, 416))],
    )
    path = save_spec(spec, tmp_path / "detection.yaml")
    assert yaml.safe_load(path.read_text())["task"] == "object_detection"
    reloaded = from_yaml(path, check_paths=False)
    assert isinstance(reloaded, DetectionSpec)
    assert reloaded.models[0].model_fields_set - {"task"} == {"name", "input_shape"}


def test_the_file_is_readable_by_a_person(tmp_path: Path):
    """Keys in declaration order rather than alphabetical, because the order the fields were
    designed in is the order they explain themselves in - `task`, then `data`, then `models`."""
    text = save_spec(rich(), tmp_path / "spec.yaml").read_text()
    assert text.index("task:") < text.index("data:") < text.index("models:")
    assert "!!python" not in text
