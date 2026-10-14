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


# ------------------------------------------------- the loss has to be over classes
# `pyplatypus#223`. The refusal these cover used to live on `InstanceModel`, which is a
# subclass of the model they are about, so the semantic task accepted `loss: mse` and ran
# with it. Nothing downstream refused it either - the loader builds the foreground mask for
# any loss that asks - and the loss it reported was *small*: measured on eight 32x32
# samples over 80 epochs, a validation loss of 0.0014 against cce's 0.0865 and a Dice of
# 0.54 against 1.00. These are at the spec layer because that is where it is cheap.
@pytest.mark.parametrize("name", ["mse", "l1", "smooth_l1"])
def test_a_continuous_loss_is_refused_as_a_models_own_loss(config, name):
    config["models"][0]["loss"] = {"name": name}
    with pytest.raises(ConfigError) as caught:
        from_dict(config, check_paths=False)
    message = str(caught.value)
    assert name in message
    assert "class channels" in message


@pytest.mark.parametrize("name", ["mse", "l1", "smooth_l1"])
def test_a_continuous_loss_is_refused_with_regions_too(config, name):
    """The other naming of the classes, and the one where the refusal is least obvious:
    with `regions` there is no background channel, so the foreground `where` would mask to
    is "any channel hot" rather than "not channel 0". Different arithmetic, same hole."""
    config["data"].pop("colormap")
    config["data"]["regions"] = [{"name": "a", "of": [1]}, {"name": "b", "of": [1, 2]}]
    config["models"][0]["loss"] = {"name": name}
    with pytest.raises(ConfigError, match="class channels"):
        from_dict(config, check_paths=False)


def test_where_everywhere_is_refused_without_being_told_it_said_foreground(config):
    """A message may not describe a setting its reader did not write. `where: foreground`
    is the default and the clause explaining it is the useful half of this refusal - but
    quoted at somebody who wrote `everywhere` it sends them to change the wrong line.

    This also records that `where: everywhere` is refused *deliberately* rather than as
    collateral: on the fixture above it reached a Dice of 0.9838, so an unactivated squared
    error is a crude surrogate and not nonsense. It is refused because `loss` is the
    objective on the classes, and an activated regression would be a new loss with its own
    name and its own measurement rather than this family, whose whole content is `where`.
    """
    config["models"][0]["loss"] = {"name": "mse", "where": "everywhere"}
    with pytest.raises(ConfigError) as caught:
        from_dict(config, check_paths=False)
    message = str(caught.value)
    assert "class channels" in message
    assert "where: foreground" not in message

    config["models"][0]["loss"] = {"name": "mse"}
    with pytest.raises(ConfigError) as caught:
        from_dict(config, check_paths=False)
    assert "where: foreground" in str(caught.value)


def test_the_refusal_is_on_the_model_that_has_the_one_hot_target(config):
    """Asked of `SegmentationModel` directly rather than through a specification, because
    *which class* carries this is the whole content of the fix: it was on the subclass, and
    a subclass cannot refuse on behalf of its parent."""
    from pydantic import ValidationError

    from pyplatypus.spec.components import CceLoss, MseLoss
    from pyplatypus.spec.models import SegmentationModel

    assert SegmentationModel(name="m", input_shape=[64, 64], loss=CceLoss()).loss.name == "cce"
    with pytest.raises(ValidationError, match="class channels"):
        SegmentationModel(name="m", input_shape=[64, 64], loss=MseLoss())
