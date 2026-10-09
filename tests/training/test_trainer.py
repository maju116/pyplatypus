"""The training loop, on data small enough to run in a second."""

import numpy as np
import pytest
import torch
from torch import nn

from pyplatypus.data import SegmentationDataset, discover
from pyplatypus.data.images import stitch
from pyplatypus.models import build_model
from pyplatypus.spec.common import Architecture
from pyplatypus.spec.components import DiceMetric, EarlyStopping, TerminateOnNaN
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.training import Trainer, make_loader
from tests.conftest import write_png


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


def test_predict_stream_yields_the_same_images_predict_stacks(loaders):
    """The streaming form is not an approximation of the stacked one: same numbers."""
    spec = model_spec(input_shape=(32, 32), splits=(2, 2))
    trainer = Trainer(build_model(spec, n_class=2), spec, device="cpu")
    stacked = trainer.predict(loaders(spec))
    streamed = list(trainer.predict_stream(loaders(spec)))
    assert len(streamed) == len(stacked) == 3
    for one, row in zip(streamed, stacked, strict=True):
        assert np.array_equal(one, row)


def test_predict_stream_stitches_tiles_that_straddle_a_batch(loaders):
    """4 tiles an image and batches of 3, so no image's tiles arrive in one batch.

    This is the whole risk in buffering by example rather than by batch, and it is why the
    buffer counts examples: a version that stitched each batch would mix two pictures.
    """
    spec = model_spec(input_shape=(32, 32), splits=(2, 2))
    trainer = Trainer(build_model(spec, n_class=2), spec, device="cpu")
    aligned = trainer.predict(make_loader(loaders(spec).dataset.base, batch_size=4))
    straddled = list(trainer.predict_stream(make_loader(loaders(spec).dataset.base, batch_size=3)))
    assert len(straddled) == 3
    for one, row in zip(straddled, aligned, strict=True):
        assert np.array_equal(one, row)


def test_predict_stream_stops_reading_when_the_caller_stops(loaders):
    """The point of the thing: one image out does not read the whole split in.

    `predict` cannot do this - it needs every batch before it returns anything - and that
    is what made it unusable on a large tiled split (#178).
    """
    spec = model_spec(input_shape=(32, 32), splits=(2, 2))
    trainer = Trainer(build_model(spec, n_class=2), spec, device="cpu")

    class Counted:
        """Counts batches handed over, so "did it read everything" is measurable."""

        def __init__(self, loader):
            self.loader = loader
            self.batches = 0

        def __iter__(self):
            for batch in self.loader:
                self.batches += 1
                yield batch

    counted = Counted(make_loader(loaders(spec).dataset.base, batch_size=1))
    first = next(iter(trainer.predict_stream(counted)))
    assert first.shape == (64, 64, 2)
    assert counted.batches == 4, "one image is 4 tiles; reading more is reading the split"

    whole = Counted(make_loader(loaders(spec).dataset.base, batch_size=1))
    list(trainer.predict_stream(whole))
    assert whole.batches == 12, "3 images x 4 tiles, so the full pass is still a full pass"


def test_predict_stream_refuses_a_partial_grid(loaders):
    """Same guard as `predict`, which now runs through this: leftover tiles are an error
    rather than a silently short last image."""
    spec = model_spec(input_shape=(32, 32), splits=(2, 2))
    trainer = Trainer(build_model(spec, n_class=2), spec, device="cpu")
    loader = make_loader(loaders(spec).dataset.base, batch_size=5, drop_last=True)
    with pytest.raises(ValueError, match="whole number of images"):
        list(trainer.predict_stream(loader))


class _FixedMask(nn.Module):
    """A model that ignores its input and always predicts the same mask.

    The point of a stub here is that the expected numbers can be written down. A real model
    on a toy fixture predicts something arbitrary, and in one attempt at this test it
    predicted the whole frame as foreground - whose skeleton is empty, because a mask with
    no border has no centreline, so clDice refused the case and the test measured nothing.
    """

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        # One parameter it never uses: `Trainer` builds an optimizer, and torch refuses an
        # empty parameter list.
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        one = self.mask.to(x.device)[None].expand(x.shape[0], -1, -1, -1)
        return torch.where(one > 0, 8.0, -8.0)


def _seamed_prediction(size: int) -> torch.Tensor:
    """A two-pixel line across the middle, with a gap exactly where two tiles meet.

    Per tile the line is intact and scores well. Reassembled it is severed, which is the
    kind of error clDice exists to see and Dice barely notices - so the two ways of
    measuring it give different numbers, and the test can tell which one happened.
    """
    mask = torch.zeros(2, size, size)
    middle = size // 2
    mask[1, middle : middle + 2, :] = 1.0
    mask[1, middle : middle + 2, middle - 1 : middle + 1] = 0.0  # the gap, at the seam
    mask[0] = 1.0 - mask[1]
    return mask


@pytest.fixture
def thin_lines(tmp_path, binary_data):
    """Two samples whose mask is a two-pixel line, which is what clDice is for.

    The shared fixtures draw a solid 40x64 block flush against three edges, and such a
    thing has no centreline to find - clDice refuses it rather than scoring it 1.0, which
    is correct and useless here. A line has a skeleton, and crossing the tile seam is the
    whole point.
    """
    root = tmp_path / "lines"
    for n in range(2):
        sample = root / f"sample_{n}"
        write_png(sample / "images" / "a.png", np.full((64, 64, 3), 10 * n, np.uint8))
        mask = np.zeros((64, 64, 3), np.uint8)
        mask[30:32, :] = 255  # across the vertical seam at column 32
        write_png(sample / "masks" / "a.png", mask)
    return discover(root, binary_data).samples


def test_a_tiled_case_is_stitched_before_a_whole_mask_metric_reads_it(thin_lines, binary_data):
    """The claim the whole change rests on, with numbers that can be written down.

    `score_cases` holds a case's tiles until its image is complete and measures clDice once
    on the reassembled pair. Measured per tile instead, a line severed exactly at the seam
    looks intact in both halves and scores higher - so asserting the reported number equals
    the stitched one *and* differs from the per-tile mean is what tells the two apart. A
    test that only checked the first would pass on either implementation.
    """
    spec = model_spec(input_shape=(32, 32), splits=(2, 2), metrics=[{"name": "cldice"}])
    prediction = torch.zeros(2, 32, 32)
    prediction[1, 30:32, :] = 1.0
    prediction[1, 30:32, 30:32] = 0.0  # the gap, at the seam each tile has an edge on
    prediction[0] = 1.0 - prediction[1]

    trainer = Trainer(_FixedMask(prediction), spec, device="cpu")
    base = SegmentationDataset(thin_lines, spec, binary_data)
    loader = make_loader(base, batch_size=spec.batch_size)
    rows = trainer.score_cases(loader, ["one"] * 4 + ["two"] * 4)
    reported = rows[0]["cldice"]

    metric = trainer.metrics["cldice"]
    targets = torch.stack([torch.as_tensor(base[i][1]).permute(2, 0, 1).float() for i in range(4)])
    predictions = prediction[None].expand(4, -1, -1, -1)
    stitched = metric.reduce(
        metric.coefficient(
            trainer._stitch_channels_first(predictions)[None],
            trainer._stitch_channels_first(targets)[None],
        )[0]
    ).item()
    per_tile = float(
        np.mean(
            [metric.reduce(metric.coefficient(predictions, targets)[i]).item() for i in range(4)]
        )
    )

    assert reported == pytest.approx(stitched, abs=1e-5)
    assert abs(reported - per_tile) > 1e-3, (
        f"stitched {stitched:.4f} and per-tile {per_tile:.4f} agree, so this fixture cannot "
        "tell the two implementations apart and the test proves nothing"
    )


def test_the_torch_stitch_matches_the_numpy_one(loaders):
    """Two implementations of one piece of arithmetic, held to each other.

    `data.images.stitch` is the tested one and works channels-last on numpy;
    `_stitch_channels_first` exists so scoring does not copy a four-megapixel mask to the
    host and back twice an image. A non-square grid and non-square tiles are in here
    because an index slip survives the square case.
    """
    for splits, tile, channels in (((2, 2), (8, 6), 3), ((4, 4), (5, 5), 2), ((3, 2), (4, 7), 1)):
        spec = model_spec(input_shape=(32, 32), splits=splits)
        trainer = Trainer(build_model(spec, n_class=2), spec, device="cpu")
        count = int(np.prod(splits))
        last = np.random.rand(count, *tile, channels).astype(np.float32)
        first = torch.as_tensor(np.moveaxis(last, -1, 1).copy())
        want = stitch(last, splits)
        got = np.moveaxis(trainer._stitch_channels_first(first).numpy(), 0, -1)
        assert np.array_equal(want, got), f"{splits} with {tile} tiles disagree"


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
