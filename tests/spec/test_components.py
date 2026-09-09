"""Losses, optimisers and callbacks: the parts that used to be typed `Any`."""

import pytest

from pyplatypus import ConfigError, from_dict


@pytest.mark.parametrize("name", [
    "iou", "dice", "cce", "cce_dice", "focal", "tversky", "focal_tversky", "combo", "lovasz",
])
def test_every_loss_is_reachable(config, name):
    config["models"][0]["loss"] = {"name": name}
    assert from_dict(config).models[0].loss.name == name


@pytest.mark.parametrize("name", [
    "adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam",
])
def test_every_optimiser_is_reachable(config, name):
    config["models"][0]["optimizer"] = {"name": name}
    assert from_dict(config).models[0].optimizer.name == name


def test_loss_parameters_are_range_checked(config):
    config["models"][0]["loss"] = {"name": "tversky", "alpha": 1.5}
    with pytest.raises(ConfigError):
        from_dict(config)


def test_tversky_betas_complement_alpha(config):
    config["models"][0]["loss"] = {"name": "tversky", "alpha": 0.3}
    assert from_dict(config).models[0].loss.beta == pytest.approx(0.7)


def test_nesterov_without_momentum_is_refused(config):
    config["models"][0]["optimizer"] = {"name": "sgd", "nesterov": True}
    with pytest.raises(ConfigError, match="momentum"):
        from_dict(config)


def test_checkpoint_requires_a_path(config):
    config["models"][0]["callbacks"] = [{"name": "model_checkpoint"}]
    with pytest.raises(ConfigError):
        from_dict(config)


def test_not_fitting_without_weights_is_refused(config):
    config["models"][0]["fit"] = False
    with pytest.raises(ConfigError, match="weights"):
        from_dict(config)


def test_loading_weights_without_fitting_is_fine(config):
    config["models"][0]["fit"] = False
    config["models"][0]["weights"] = "dsbowl2018"
    assert from_dict(config).models[0].weights == "dsbowl2018"


def test_deep_supervision_needs_depth(config):
    config["models"][0]["blocks"] = 1
    config["models"][0]["input_shape"] = [32, 32]
    config["models"][0]["architecture"] = "u_net_plus_plus"
    config["models"][0]["deep_supervision"] = True
    with pytest.raises(ConfigError, match="at least 2 blocks"):
        from_dict(config)


def test_deep_supervision_only_makes_sense_for_the_nested_architecture(config):
    """It reads the intermediate nodes X[0][j], which only u_net_plus_plus produces."""
    config["models"][0]["deep_supervision"] = True
    with pytest.raises(ConfigError, match="u_net_plus_plus"):
        from_dict(config)


def test_deep_supervision_is_fine_on_u_net_plus_plus(config):
    config["models"][0]["architecture"] = "u_net_plus_plus"
    config["models"][0]["deep_supervision"] = True
    assert from_dict(config).models[0].deep_supervision
