"""The error messages are a user-facing feature; they get tested like one."""

import pytest

from pyplatypus import ConfigError, from_dict


def test_unknown_key_is_not_silently_ignored(config):
    """A typo that quietly does nothing is the failure mode this package exists to avoid."""
    config["models"][0]["filterz"] = 32
    with pytest.raises(ConfigError) as caught:
        from_dict(config)
    assert "filterz" in str(caught.value)


def test_unknown_loss_is_named(config):
    config["models"][0]["loss"] = {"name": "focaal"}
    with pytest.raises(ConfigError) as caught:
        from_dict(config)
    assert "focaal" in str(caught.value)


def test_augmentation_typo_gets_a_suggestion(config):
    config["models"][0]["augmentation"] = [{"name": "horizontalflip"}]
    with pytest.raises(ConfigError) as caught:
        from_dict(config)
    assert "HorizontalFlip" in str(caught.value)


def test_augmentation_names_come_from_the_installed_albumentations(config):
    """A hardcoded list drifts. `Flip` existed in albumentations 1.x and is gone in 2.x;
    reading the names off the installed package is what keeps the spec honest."""
    import albumentations as A

    from pyplatypus.spec.components import available_transforms

    known = available_transforms()
    assert "HorizontalFlip" in known
    assert known <= {n for n in dir(A) if n[:1].isupper()}


def test_a_real_albumentations_transform_is_accepted(config):
    config["models"][0]["augmentation"] = [{"name": "D4", "params": {"p": 0.5}}]
    assert from_dict(config).models[0].augmentation[0].name == "D4"


def test_every_problem_is_reported_not_just_the_first(config):
    config["models"][0]["filters"] = -1
    config["models"][0]["epochs"] = 0
    config["models"][0]["dropout"] = 2.0
    with pytest.raises(ConfigError) as caught:
        from_dict(config)
    assert len(caught.value.problems) >= 3


def test_location_reads_like_a_path(config):
    config["models"][0]["filters"] = -1
    with pytest.raises(ConfigError) as caught:
        from_dict(config)
    assert any(p["where"] == "models[0].filters" for p in caught.value.problems)


def test_error_survives_as_plain_data(config):
    """This is what crosses the bridge into R, so it must be dicts and strings."""
    config["models"][0]["filters"] = -1
    with pytest.raises(ConfigError) as caught:
        from_dict(config)
    payload = caught.value.to_dict()
    assert payload["kind"] == "config_error"
    assert isinstance(payload["problems"], list)
    assert all(isinstance(value, str) for p in payload["problems"] for value in p.values())


def test_non_mapping_config_is_refused():
    with pytest.raises(ConfigError, match="mapping"):
        from_dict(["not", "a", "mapping"])
