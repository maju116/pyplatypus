"""Per-case scores, and the arithmetic that makes them the right number.

A ratio of sums is not the mean of ratios. Whenever a case arrives in pieces - tiles now,
slices of a volume later - the score has to be assembled from the pieces' overlap counts
rather than from the pieces' scores. The test that pins this is the one worth keeping.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyplatypus import Engine, from_dict, summarise_cases
from pyplatypus.engine import EngineError
from pyplatypus.errors import ConfigError
from pyplatypus.objectives.metrics import Dice
from tests.conftest import metric_configs, write_png


@pytest.fixture
def patients(tmp_path):
    """Four patients, three slices each, with a square to find."""
    root = tmp_path / "data"
    for patient in range(4):
        for index in range(3):
            key = f"patient{patient:02d}_slice{index:03d}"
            mask = np.zeros((32, 32, 3), np.uint8)
            mask[6 + patient : 18 + patient, 8:24] = 255
            write_png(root / key / "images" / "a.png", mask // 2)
            write_png(root / key / "masks" / "a.png", mask)
    return root


def spec_for(root, **model):
    block = {
        "name": "tiny",
        "input_shape": [32, 32],
        "blocks": 2,
        "filters": 4,
        "batch_size": 2,
        "epochs": 1,
        "metrics": [{"name": "dice"}],
    }
    block.update(model)
    return from_dict(
        {
            "task": "semantic_segmentation",
            "data": {
                "train_path": str(root),
                "validation_path": str(root),
                "colormap": [[0, 0, 0], [255, 255, 255]],
                "shuffle": False,
            },
            "models": [block],
        }
    )


def test_one_row_per_case_with_the_metrics_asked_for(patients):
    engine = Engine(spec_for(patients), device="cpu")
    engine.fit()
    rows = engine.evaluate_cases("tiny")

    assert len(rows) == 12
    assert [row["case"] for row in rows] == sorted(row["case"] for row in rows)
    assert all("dice" in row and 0.0 <= row["dice"] <= 1.0 for row in rows)


def test_grouping_reports_patients_instead_of_slices(patients):
    engine = Engine(spec_for(patients), device="cpu")
    engine.fit()
    rows = engine.evaluate_cases("tiny", group_by=r"^(patient\d+)_")

    assert len(rows) == 4
    assert {row["group"] for row in rows} == {f"patient{n:02d}" for n in range(4)}
    assert "case" not in rows[0]


def test_tiles_are_summed_into_the_whole_image_not_averaged(patients):
    """The arithmetic claim, checked against Dice computed on the whole mask at once.

    A model that tiles sees a quarter of the image at a time. Averaging the four tiles'
    Dice gives a different number from the image's Dice - usually a worse one, because a
    tile containing a sliver of the object is easy to score badly on.
    """
    engine = Engine(spec_for(patients, splits=[2, 2]), device="cpu")
    engine.fit()
    rows = engine.evaluate_cases("tiny")
    reported = {row["case"]: row["dice"] for row in rows}

    run = engine.runs["tiny"]
    dataset = engine.dataset(run.spec, "validation")
    metric = Dice(smooth=1.0)
    run.model.eval()

    for index, sample in enumerate(dataset.samples):
        tiles = dataset.tiles_per_sample
        images, targets = [], []
        for tile_index in range(tiles):
            image, target = dataset[index * tiles + tile_index]
            images.append(image)
            targets.append(target)
        batch_x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()
        batch_y = torch.from_numpy(np.stack(targets)).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            logits = run.model(batch_x)
        hard = logits.argmax(dim=1)

        # Reassemble the tiles into whole images - the 2x2 grid came out row-major - and
        # score once, which is what the reported number must equal.
        grid_prediction = torch.cat(
            [torch.cat([hard[0], hard[1]], dim=1), torch.cat([hard[2], hard[3]], dim=1)], dim=0
        )
        grid_target = torch.cat(
            [
                torch.cat([batch_y[0], batch_y[1]], dim=2),
                torch.cat([batch_y[2], batch_y[3]], dim=2),
            ],
            dim=1,
        )
        whole = metric(
            torch.nn.functional.one_hot(grid_prediction, 2).permute(2, 0, 1)[None].float(),
            grid_target[None],
        ).item()

        assert reported[sample.key] == pytest.approx(whole, abs=1e-5)


# Every one of the three tests below was written for `cldice` and read as being about
# whole-mask metrics; `surface_dice` shipped in `0.8.0a26` and none of them saw it, which
# is how the specification layer came to accept a callback the trainer would not feed.
# They are parametrised over the metrics that say they are whole-mask rather than over a
# list written here, so the next one of these is covered by arriving. §4.
WHOLE_MASK = metric_configs(whole_mask=True)
WHOLE_MASK_IDS = [block["name"] for block in WHOLE_MASK]


@pytest.mark.parametrize("metric", WHOLE_MASK, ids=WHOLE_MASK_IDS)
def test_a_tiled_run_may_ask_for_a_whole_mask_metric(patients, metric):
    """It used to be refused outright. The refusal was right about the arithmetic and
    wrong about the remedy: the tiles are reassembled before the metric is measured, so
    the combination is now the supported way to ask."""
    engine = Engine(spec_for(patients, splits=[2, 2], metrics=[metric]))
    assert engine.spec.models[0].splits == (2, 2)


@pytest.mark.parametrize("metric", WHOLE_MASK, ids=WHOLE_MASK_IDS)
def test_a_tiled_epoch_leaves_out_the_metric_it_cannot_measure(patients, metric):
    """An epoch reads its metrics per batch, and per batch means per tile. Rather than
    report a per-tile skeleton or a per-tile boundary, a tiled run omits it from the
    history - and says so by not offering it to callbacks either."""
    name = metric["name"]
    tiled = Engine(
        spec_for(patients, splits=[2, 2], metrics=[{"name": "dice"}, metric]),
        device="cpu",
    )
    record = tiled.fit()["tiny"].records[-1]
    assert "val_dice" in record
    assert f"val_{name}" not in record
    assert f"train_{name}" not in record

    # Untiled, an example is a whole mask, so the epoch can and does report it.
    whole = Engine(spec_for(patients, metrics=[{"name": "dice"}, metric]), device="cpu")
    assert f"val_{name}" in whole.fit()["tiny"].records[-1]


@pytest.mark.parametrize("metric", WHOLE_MASK, ids=WHOLE_MASK_IDS)
def test_a_callback_cannot_watch_what_a_tiled_epoch_does_not_produce(patients, metric):
    """The pairing is refused when the specification is read, not mid-run, and the message
    lists what is available - which is the existing monitor check doing the work rather
    than a second validator repeating it.

    What it costs when the two layers disagree is the reason this is parametrised:
    `EarlyStopping` indexes `state.logs[self.monitor]`, so an accepted configuration that
    the epoch does not feed raises `KeyError` after the first epoch has been paid for.
    """
    watched = f"val_{metric['name']}"
    with pytest.raises(ConfigError, match=watched):
        spec_for(
            patients,
            splits=[2, 2],
            metrics=[{"name": "dice"}, metric],
            callbacks=[{"name": "early_stopping", "monitor": watched, "patience": 2}],
        )

    # Untiled it is a perfectly good thing to watch.
    spec_for(
        patients,
        metrics=[metric],
        callbacks=[{"name": "early_stopping", "monitor": watched, "patience": 2}],
    )


def test_the_mean_of_case_scores_is_close_to_the_split_score(patients):
    """Sanity, not equality: the split score averages over batches, so they differ a
    little. A large gap would mean one of the two is measuring something else."""
    engine = Engine(spec_for(patients), device="cpu")
    engine.fit()
    per_case = np.mean([row["dice"] for row in engine.evaluate_cases("tiny")])
    whole_split = engine.evaluate()[0]["dice"]
    assert per_case == pytest.approx(whole_split, abs=0.05)


def test_a_model_without_metrics_says_so(patients):
    engine = Engine(spec_for(patients, metrics=[]), device="cpu")
    engine.fit()
    with pytest.raises(EngineError, match="no metrics"):
        engine.evaluate_cases("tiny")


def test_an_unknown_model_lists_the_ones_there_are(patients):
    engine = Engine(spec_for(patients), device="cpu")
    engine.fit()
    with pytest.raises(EngineError, match="no model called"):
        engine.evaluate_cases("nope")


def test_a_group_pattern_that_matches_nothing_is_refused(patients):
    engine = Engine(spec_for(patients), device="cpu")
    engine.fit()
    with pytest.raises(Exception, match="does not match"):
        engine.evaluate_cases("tiny", group_by=r"^(subject\d+)_")


# ------------------------------------------------------------------- summarising
def test_the_summary_reports_the_distribution_and_the_worst_case():
    rows = [
        {"case": "a", "dice": 0.9, "iou": 0.8},
        {"case": "b", "dice": 0.5, "iou": 0.4},
        {"case": "c", "dice": 0.7, "iou": 0.6},
    ]
    summary = {row["metric"]: row for row in summarise_cases(rows)}

    assert summary["dice"]["n"] == 3
    assert summary["dice"]["mean"] == pytest.approx(0.7)
    assert summary["dice"]["median"] == pytest.approx(0.7)
    assert summary["dice"]["min"] == pytest.approx(0.5)
    assert summary["dice"]["sd"] == pytest.approx(0.2, abs=1e-6)
    # Naming the worst case is the point: a mean of 0.7 does not tell you to go and look
    # at patient b.
    assert summary["dice"]["worst_case"] == "b"
    assert summary["iou"]["worst_case"] == "b"


def test_the_spread_of_one_case_is_unknown_rather_than_zero():
    summary = summarise_cases([{"case": "only", "dice": 0.8}])[0]
    assert summary["n"] == 1
    assert summary["sd"] is None


def test_grouped_rows_are_summarised_by_group():
    summary = summarise_cases(
        [
            {"group": "patient01", "dice": 0.4},
            {"group": "patient02", "dice": 0.8},
        ]
    )[0]
    assert summary["worst_group"] == "patient01"


def test_summarising_nothing_is_an_error():
    with pytest.raises(EngineError, match="no case scores"):
        summarise_cases([])


def test_rows_without_metrics_are_an_error():
    with pytest.raises(EngineError, match="no metric columns"):
        summarise_cases([{"case": "a"}])


# ------------------------------------------- the worst case, from the engine's side
def test_the_history_carries_the_worst_case(patients):
    """The engine knows the names the validation loader serves, so the epoch can report it."""
    engine = Engine(spec_for(patients), device="cpu")
    histories = engine.fit()
    history = histories["tiny"]

    assert "val_dice_worst" in history.columns
    for record in history.records:
        assert record["val_dice_worst"] <= record["val_dice"] + 1e-9


def test_the_history_agrees_with_evaluate_cases_at_the_last_epoch(patients):
    """Two reports of one quantity, which is what `_PerCase` exists to keep from drifting."""
    engine = Engine(spec_for(patients), device="cpu")
    history = engine.fit()["tiny"]
    rows = engine.evaluate_cases("tiny")

    assert history.records[-1]["val_dice_worst"] == pytest.approx(
        min(row["dice"] for row in rows), abs=1e-6
    )


def test_a_callback_may_watch_the_worst_case(patients):
    """It is in the logs, so the validator has to admit it - and it is validation only,
    because the training loader is shuffled and a case has no identity there."""
    spec = spec_for(patients, callbacks=[{"name": "early_stopping", "monitor": "val_dice_worst"}])
    engine = Engine(spec, device="cpu")
    assert engine.fit()["tiny"] is not None

    with pytest.raises(ConfigError, match="train_dice_worst"):
        spec_for(patients, callbacks=[{"name": "early_stopping", "monitor": "train_dice_worst"}])
