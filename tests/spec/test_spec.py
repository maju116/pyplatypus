"""What a good spec accepts and what a bad one says about itself."""

import pytest

from pyplatypus import ConfigError, from_dict
from pyplatypus.spec import Architecture, PlatypusSpec


def test_minimal_config_builds(config):
    spec = from_dict(config)
    assert isinstance(spec, PlatypusSpec)
    assert spec.models[0].name == "unet"
    assert spec.models[0].architecture is Architecture.U_NET
    assert spec.rank == 2


def test_defaults_are_sensible(config):
    model = from_dict(config).models[0]
    assert model.loss.name == "cce"
    assert [m.name for m in model.metrics] == ["iou"]
    assert model.optimizer.name == "adam"
    assert model.optimizer.learning_rate == pytest.approx(1e-3)


def test_n_class_comes_from_the_colormap(config):
    config["data"]["colormap"] = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]
    config["models"][0]["n_class"] = 3
    assert from_dict(config).data.n_class == 3


def test_class_count_must_agree_with_colormap(config):
    config["models"][0]["n_class"] = 5
    with pytest.raises(ConfigError) as caught:
        from_dict(config)
    assert "colormap defines 2 classes" in str(caught.value)


def test_duplicate_model_names_rejected(config, model_block):
    config["models"] = [dict(model_block), dict(model_block)]
    with pytest.raises(ConfigError, match="unique"):
        from_dict(config)


def test_duplicate_colours_rejected(config):
    config["data"]["colormap"] = [[0, 0, 0], [0, 0, 0]]
    with pytest.raises(ConfigError, match="distinct"):
        from_dict(config)


def test_missing_paths_are_reported(config):
    config["data"]["train_path"] = "/no/such/place"
    with pytest.raises(ConfigError) as caught:
        from_dict(config)
    assert "does not exist" in str(caught.value)


def test_paths_can_be_left_unchecked(config):
    """R may build a spec on a machine that does not hold the data."""
    config["data"]["train_path"] = "/no/such/place"
    assert from_dict(config, check_paths=False).data.train_path == "/no/such/place"
