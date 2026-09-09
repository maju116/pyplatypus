"""YAML is just another constructor. Same object, or the contract is broken."""

from pathlib import Path

import pytest
import yaml

from pyplatypus import ConfigError, from_dict, from_yaml

FIXTURE = "tests/fixtures/experiment.yaml"


@pytest.fixture
def yaml_file(tmp_path, data_block):
    text = (
        Path(FIXTURE)
        .read_text()
        .replace("PLACEHOLDER_TRAIN", data_block["train_path"])
        .replace("PLACEHOLDER_VALID", data_block["validation_path"])
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(text)
    return path


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
