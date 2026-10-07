"""The training loop, on data small enough to run in a second."""

import numpy as np
import pytest
import torch

from pyplatypus.data import SegmentationDataset, discover
from pyplatypus.models import build_model
from pyplatypus.spec.common import Architecture
from pyplatypus.spec.components import DiceMetric, EarlyStopping, TerminateOnNaN
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.training import Trainer, make_loader


def model_spec(**overrides):
    base = {
        "name": "m",
        "input_shape": (32, 32),
        "channels": 3,
        "blocks": 2,
        "filters": 4,
        "batch_size": 2,
        "epochs": 2,
        "metrics": [DiceMetric()],
    }
    return SegmentationModel(**{**base, **overrides})


@pytest.fixture
def loaders(nested_root, binary_data):
    samples = discover(nested_root, binary_data).samples

    def build(spec, **kwargs):
        base = SegmentationDataset(samples, spec, binary_data, **kwargs)
        return make_loader(base, batch_size=spec.batch_size)

    return build


def test_fit_returns_one_record_per_epoch(loaders):
    spec = model_spec(epochs=3)
    trainer = Trainer(build_model(spec, n_class=2), spec, device="cpu")
    history = trainer.fit(loaders(spec), loaders(spec))
    assert len(history) == 3
    assert [r["epoch"] for r in history.records] == [1, 2, 3]


def test_history_carries_losses_metrics_and_timing(loaders):
    spec = model_spec()
    history = Trainer(build_model(spec, n_class=2), spec, device="cpu").fit(
        loaders(spec), loaders(spec)
    )
    assert set(history.columns) >= {
        "epoch",
        "train_loss",
        "train_dice",
        "val_loss",
        "val_dice",
        "seconds",
        "learning_rate",
    }


def test_training_actually_reduces_the_loss(loaders):
    spec = model_spec(epochs=12, filters=8)
    torch.manual_seed(0)
    history = Trainer(build_model(spec, n_class=2), spec, device="cpu").fit(loaders(spec))
    assert history.records[-1]["train_loss"] < history.records[0]["train_loss"]


def test_validation_is_optional(loaders):
    spec = model_spec()
    history = Trainer(build_model(spec, n_class=2), spec, device="cpu").fit(loaders(spec))
    assert "val_loss" not in history.columns


def test_best_picks_the_lowest_loss_and_the_highest_metric(loaders):
    spec = model_spec(epochs=3)
    history = Trainer(build_model(spec, n_class=2), spec, device="cpu").fit(
        loaders(spec), loaders(spec)
    )
    assert history.best("val_loss") == min(history.records, key=lambda r: r["val_loss"])
    assert history.best("val_dice") == max(history.records, key=lambda r: r["val_dice"])


def test_callbacks_can_cut_training_short(loaders):
    spec = model_spec(epochs=20, callbacks=[EarlyStopping(patience=1, min_delta=1e9)])
    history = Trainer(build_model(spec, n_class=2), spec, device="cpu").fit(
        loaders(spec), loaders(spec)
    )
    assert len(history) < 20
    assert "early stopping" in history.stop_reason


def test_a_nan_loss_stops_the_run(loaders):
    """A NaN never recovers; carrying on just burns GPU hours."""
    spec = model_spec(epochs=5, callbacks=[TerminateOnNaN()])
    trainer = Trainer(build_model(spec, n_class=2), spec, device="cpu")
    trainer.loss_fn = lambda logits, target: logits.sum() * float("nan")
    history = trainer.fit(loaders(spec), loaders(spec))
    assert len(history) == 1 and "nan" in history.stop_reason


def test_deep_supervision_trains_on_every_output(loaders):
    """The loss averages over all depths; the metrics score only the final prediction."""
    spec = model_spec(architecture=Architecture.U_NET_PLUS_PLUS, deep_supervision=True)
    history = Trainer(build_model(spec, n_class=2), spec, device="cpu").fit(
        loaders(spec), loaders(spec)
    )
    assert len(history) == 2
    assert 0.0 <= history.records[-1]["val_dice"] <= 1.0


def test_evaluate_scores_without_training(loaders):
    spec = model_spec()
    scores = Trainer(build_model(spec, n_class=2), spec, device="cpu").evaluate(loaders(spec))
    assert set(scores) == {"val_loss", "val_dice"}


def test_predict_returns_channels_last_probabilities(loaders):
    spec = model_spec()
    predictions = Trainer(build_model(spec, n_class=2), spec, device="cpu").predict(loaders(spec))
    assert predictions.shape == (3, 32, 32, 2)
    assert np.allclose(predictions.sum(axis=-1), 1.0, atol=1e-5)


def test_predict_reassembles_tiles_into_whole_images(loaders):
    """The capability the old package lacked: an image cut into a grid comes back whole."""
    spec = model_spec(input_shape=(32, 32), splits=(2, 2))
    predictions = Trainer(build_model(spec, n_class=2), spec, device="cpu").predict(loaders(spec))
    assert predictions.shape == (3, 64, 64, 2)  # 3 images at 2x2 tiles of 32x32


def test_predict_refuses_a_partial_grid(loaders):
    """3 images x 4 tiles = 12; a batch of 5 with drop_last leaves 10, which is not a
    whole number of images. Stitching that would silently mix two pictures together."""
    spec = model_spec(input_shape=(32, 32), splits=(2, 2))
    trainer = Trainer(build_model(spec, n_class=2), spec, device="cpu")
    loader = make_loader(loaders(spec).dataset.base, batch_size=5, drop_last=True)
    with pytest.raises(ValueError, match="whole number of images"):
        trainer.predict(loader)


# --- the distance map's route through the data path ----------------------------------------


def test_the_loader_carries_a_distance_map_only_when_the_loss_asks(config, nested_root):
    """Two items in the batch or three, and the loader decides from the specification.

    The transform costs more than an epoch of a small 3D model, so everyone paying for it
    would be the wrong default; and a loader that always produced it would make the
    trainer's unpacking a lie.
    """
    from pyplatypus import build_engine
    from pyplatypus.spec.loader import from_dict

    config["data"]["train_path"] = str(nested_root)
    config["data"]["validation_path"] = str(nested_root)

    config["models"][0]["loss"] = {"name": "dice"}
    engine = build_engine(from_dict(config), device="cpu")
    plain = next(iter(engine.loader(engine.spec.models[0], "train")))
    assert len(plain) == 2

    config["models"][0]["loss"] = {"name": "boundary", "region": {"name": "dice"}}
    engine = build_engine(from_dict(config), device="cpu")
    batch = next(iter(engine.loader(engine.spec.models[0], "train")))
    assert len(batch) == 3
    _, mask, distance = batch
    assert distance.shape == mask.shape
    # Negative somewhere and positive somewhere: a map that is all one sign is a mask with
    # no boundary, and this fixture has one.
    assert float(distance.min()) < 0 < float(distance.max())
