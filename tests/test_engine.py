"""Many models from one spec - the reason the YAML path exists at all."""

import pytest
import torch

from pyplatypus import Engine, from_dict
from pyplatypus.engine import EngineError


@pytest.fixture
def two_model_config(nested_root):
    return {
        "data": {
            "train_path": str(nested_root),
            "validation_path": str(nested_root),
            "colormap": [[0, 0, 0], [255, 255, 255]],
            "shuffle": False,
        },
        "models": [
            {"name": "unet", "input_shape": [32, 32], "n_class": 2, "blocks": 2,
             "filters": 4, "batch_size": 2, "epochs": 2,
             "metrics": [{"name": "dice"}]},
            {"name": "linknet", "architecture": "linknet", "input_shape": [32, 32],
             "n_class": 2, "blocks": 2, "filters": 4, "batch_size": 2, "epochs": 2,
             "metrics": [{"name": "dice"}]},
        ],
    }


def test_fit_trains_every_model_in_the_spec(two_model_config):
    engine = Engine(from_dict(two_model_config), device="cpu")
    histories = engine.fit()
    assert set(histories) == {"unet", "linknet"}
    assert all(len(h) == 2 for h in histories.values())


def test_the_evaluation_table_has_one_row_per_model(two_model_config):
    engine = Engine(from_dict(two_model_config), device="cpu")
    engine.fit()
    table = engine.evaluate()
    assert [row["model"] for row in table] == ["unet", "linknet"]
    assert {row["architecture"] for row in table} == {"u_net", "linknet"}
    for row in table:
        assert row["parameters"] > 0
        assert row["epochs_run"] == 2
        assert 0.0 <= row["dice"] <= 1.0


def test_the_table_is_plain_data_for_the_trip_into_r(two_model_config):
    engine = Engine(from_dict(two_model_config), device="cpu")
    engine.fit()
    for row in engine.evaluate():
        for value in row.values():
            assert value is None or isinstance(value, (str, int, float))


def test_best_model_reads_the_table(two_model_config):
    engine = Engine(from_dict(two_model_config), device="cpu")
    engine.fit()
    assert engine.best_model("dice") in {"unet", "linknet"}
    assert engine.best_model("loss") in {"unet", "linknet"}   # same loss, so comparable


def test_ranking_by_loss_is_refused_when_the_losses_differ(two_model_config):
    """A Focal-Tversky of 0.05 is not better than a CCE-Dice of 0.14; it is not even the
    same question. Caught in the first real two-model run on the Data Science Bowl."""
    two_model_config["models"][0]["loss"] = {"name": "cce_dice"}
    two_model_config["models"][1]["loss"] = {"name": "focal_tversky"}
    engine = Engine(from_dict(two_model_config), device="cpu")
    engine.fit()
    with pytest.raises(EngineError, match="different losses"):
        engine.best_model("loss")
    assert engine.best_model("dice") in {"unet", "linknet"}   # metrics stay comparable


def test_the_table_names_the_loss_each_model_used(two_model_config):
    two_model_config["models"][1]["loss"] = {"name": "focal"}
    engine = Engine(from_dict(two_model_config), device="cpu")
    engine.fit()
    assert [row["loss_function"] for row in engine.evaluate()] == ["cce", "focal"]


def test_ranking_on_an_unknown_column_lists_the_real_ones(two_model_config):
    engine = Engine(from_dict(two_model_config), device="cpu")
    engine.fit()
    with pytest.raises(EngineError, match="no column"):
        engine.best_model("f1")


def test_evaluating_before_fitting_says_so(two_model_config):
    engine = Engine(from_dict(two_model_config), device="cpu")
    with pytest.raises(EngineError, match="call fit"):
        engine.evaluate()


def test_predicting_from_an_unknown_model_names_the_known_ones(two_model_config):
    engine = Engine(from_dict(two_model_config), device="cpu")
    engine.fit()
    with pytest.raises(EngineError, match="unet"):
        engine.predict("nonexistent", split="validation")


def test_predictions_come_back_at_source_size(two_model_config):
    engine = Engine(from_dict(two_model_config), device="cpu")
    engine.fit()
    predictions = engine.predict("unet", split="validation")
    assert predictions.shape == (3, 32, 32, 2)


def test_a_model_can_load_weights_and_skip_training(two_model_config, tmp_path):
    """fit: false plus weights - the 'I already trained this, just use it' path."""
    engine = Engine(from_dict(two_model_config), device="cpu")
    engine.fit()
    checkpoint = tmp_path / "unet.pt"
    torch.save(engine.runs["unet"].model.state_dict(), checkpoint)

    two_model_config["models"] = [dict(two_model_config["models"][0],
                                       fit=False, weights=str(checkpoint))]
    reloaded = Engine(from_dict(two_model_config), device="cpu")
    histories = reloaded.fit()
    assert len(histories["unet"]) == 0
    assert reloaded.runs["unet"].trained is False
    original = engine.runs["unet"].model.state_dict()
    restored = reloaded.runs["unet"].model.state_dict()
    assert set(original) == set(restored)
    assert all(torch.allclose(original[k].float(), restored[k].cpu().float())
               for k in original)
    assert reloaded.evaluate()[0]["dice"] >= 0.0


def test_a_missing_checkpoint_says_the_registry_is_not_wired_up(two_model_config):
    two_model_config["models"] = [dict(two_model_config["models"][0],
                                       fit=False, weights="dsbowl2018")]
    engine = Engine(from_dict(two_model_config), device="cpu")
    with pytest.raises(EngineError, match="registry"):
        engine.fit()


def test_validation_data_is_never_augmented(two_model_config):
    """Measuring a model on distorted images measures the distortion."""
    two_model_config["models"][0]["augmentation"] = [
        {"name": "HorizontalFlip", "params": {"p": 1.0}}
    ]
    engine = Engine(from_dict(two_model_config), device="cpu")
    spec = engine.spec.models[0]
    assert engine.dataset(spec, "train", augmented=True).augmenter is not None
    assert engine.dataset(spec, "validation").augmenter is None


def test_models_may_use_different_input_sizes(two_model_config):
    """Each model gets its own pipeline, so one can train at 32 and another at 64."""
    two_model_config["models"][1]["input_shape"] = [64, 64]
    engine = Engine(from_dict(two_model_config), device="cpu")
    engine.fit()
    assert engine.predict("unet", split="validation").shape[1:3] == (32, 32)
    assert engine.predict("linknet", split="validation").shape[1:3] == (64, 64)


def test_a_3d_spec_is_refused_with_a_useful_message(two_model_config):
    two_model_config["models"] = [dict(two_model_config["models"][0],
                                       input_shape=[32, 32, 32])]
    with pytest.raises(EngineError, match="v0.1"):
        Engine(from_dict(two_model_config), device="cpu")
