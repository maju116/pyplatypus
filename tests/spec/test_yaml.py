"""YAML is just another constructor. Same object, or the contract is broken."""

from pathlib import Path

import pytest
import yaml

from pyplatypus import ConfigError, from_dict, from_yaml

FIXTURE = "tests/fixtures/experiment.yaml"
DETECTION_FIXTURE = "tests/fixtures/detection.yaml"


def _written(fixture, tmp_path, data_block, name):
    text = (
        Path(fixture)
        .read_text()
        .replace("PLACEHOLDER_TRAIN", data_block["train_path"])
        .replace("PLACEHOLDER_VALID", data_block["validation_path"])
    )
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture
def yaml_file(tmp_path, data_block):
    return _written(FIXTURE, tmp_path, data_block, "experiment.yaml")


@pytest.fixture
def detection_yaml_file(tmp_path, data_block):
    return _written(DETECTION_FIXTURE, tmp_path, data_block, "detection.yaml")


def test_yaml_loads(yaml_file):
    spec = from_yaml(yaml_file)
    assert [m.name for m in spec.models] == ["unet", "unetpp"]
    assert spec.models[0].loss.name == "focal"
    assert spec.models[0].loss.gamma == pytest.approx(2.0)
    assert spec.models[1].loss.alpha == pytest.approx(0.7)


def test_yaml_and_dict_produce_the_same_object(yaml_file):
    """The whole 'kompatybilny pipeline' requirement, as a test."""
    from_file = from_yaml(yaml_file)
    from_code = from_dict(yaml.safe_load(yaml_file.read_text()))
    assert from_file == from_code
    assert from_file.to_dict() == from_code.to_dict()


def test_round_trip_through_plain_data(yaml_file):
    """R receives to_dict() and may hand it straight back."""
    spec = from_yaml(yaml_file)
    assert from_dict(spec.to_dict()) == spec


def test_missing_file_says_so(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        from_yaml(tmp_path / "nope.yaml")


def test_broken_yaml_says_so(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("data: {unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        from_yaml(path)


def test_empty_yaml_says_so(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(ConfigError, match="empty"):
        from_yaml(path)


# --- the same file format, the other task -----------------------------------------------

def test_a_detection_experiment_reads_from_the_same_file_format(detection_yaml_file):
    """One format, two tasks. The settings in this file were command-line flags during the
    BCCD run, which is what "detection is in the spec" has to mean to be worth anything."""
    from pyplatypus import DetectionSpec

    spec = from_yaml(detection_yaml_file)
    assert isinstance(spec, DetectionSpec)
    assert spec.data.classes == ["RBC", "WBC", "Platelets"]
    model = spec.models[0]
    assert (model.name, model.epochs, model.anchors) == ("bccd", 150, None)
    assert model.score_threshold == pytest.approx(0.01)
    assert model.augmentation[0].name == "HorizontalFlip"
    assert model.callbacks[0].patience == 25


def test_detection_yaml_and_dict_produce_the_same_object(detection_yaml_file):
    from_file = from_yaml(detection_yaml_file)
    from_code = from_dict(yaml.safe_load(detection_yaml_file.read_text()))
    assert from_file == from_code
    assert from_file.to_dict() == from_code.to_dict()


def test_a_detection_specs_dict_round_trips(detection_yaml_file):
    """`to_dict` is how a spec reaches R, so what comes out has to go back in - including
    `task`, which the dict has to carry or the trip back would land on segmentation."""
    spec = from_yaml(detection_yaml_file)
    payload = spec.to_dict()
    assert payload["task"] == "object_detection"
    assert from_dict(payload, check_paths=False) == spec
