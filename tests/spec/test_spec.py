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
    assert from_dict(config).data.n_class == 3


def test_a_model_cannot_declare_a_class_count(config):
    """It used to, and the spec needed a validator to catch it disagreeing with its own
    data. The data decides how many classes there are; a model that also said so was a
    second place for one fact to live, and the only thing it could add was a mismatch."""
    config["models"][0]["n_class"] = 5
    with pytest.raises(ConfigError, match="Extra inputs are not permitted"):
        from_dict(config)


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


# ------------------------------------------------------- naming the classes
def test_labels_name_the_classes_for_volumes(config):
    """Volumes label with numbers, not colours, so the spec has to accept both ways."""
    config["data"].pop("colormap")
    config["data"]["labels"] = [0, 1, 4]
    spec = from_dict(config, check_paths=False)
    assert spec.data.n_class == 3
    assert spec.data.label_map


def test_a_colormap_still_means_pictures(config):
    spec = from_dict(config, check_paths=False)
    assert not spec.data.label_map
    assert spec.data.n_class == 2


def test_giving_both_a_colormap_and_labels_is_refused(config):
    # Not a convenience worth having: two sources for the number of classes is one of them
    # going stale.
    config["data"]["labels"] = [0, 1]
    with pytest.raises(ConfigError, match="exactly one of"):
        from_dict(config, check_paths=False)


def test_giving_neither_is_refused(config):
    config["data"].pop("colormap")
    with pytest.raises(ConfigError, match="exactly one of"):
        from_dict(config, check_paths=False)


def test_labels_must_be_distinct(config):
    config["data"].pop("colormap")
    config["data"]["labels"] = [1, 1]
    with pytest.raises(ConfigError, match="distinct"):
        from_dict(config, check_paths=False)


# ------------------------------------------------------------ window naming
def test_the_window_is_called_window_and_still_answers_to_its_old_name(config):
    """`dicom_window` was named when DICOM was the only format that needed it; NIfTI needs
    the same thing. The old key keeps working, because a released R package sends it."""
    config["data"]["window"] = "lung"
    assert from_dict(config, check_paths=False).data.window == "lung"

    config["data"].pop("window")
    config["data"]["dicom_window"] = "lung"
    assert from_dict(config, check_paths=False).data.window == "lung"


def test_giving_the_window_under_both_names_is_refused(config):
    config["data"]["window"] = "lung"
    config["data"]["dicom_window"] = "bone"
    with pytest.raises(ConfigError):
        from_dict(config, check_paths=False)
