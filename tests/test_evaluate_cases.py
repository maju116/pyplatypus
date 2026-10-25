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
from pyplatypus.training.torch_data import make_loader
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


def test_the_split_score_is_exactly_the_mean_of_its_own_case_scores(patients):
    """Equality, not sanity, which is the whole of `maju116/rplatypus#196`.

    This test read `abs=0.05` and its docstring said the two "differ a little, because the
    split score averages over batches". Both halves were wrong about this fixture: 12
    samples at `batch_size` 2 is six full batches, so the two poolings were already
    identical here and the tolerance was hiding nothing. The case the prose described was
    never in the fixture - §4's commonest shape - and the test below adds it.

    The consequence this pins is the one the issue was filed for: a reader who computes
    `mean(evaluate_cases()$dice)` and a reader who asks `evaluate()` for the split's figure
    must get the same number.
    """
    engine = Engine(spec_for(patients), device="cpu")
    engine.fit()
    per_case = np.mean([row["dice"] for row in engine.evaluate_cases("tiny")])
    whole_split = engine.evaluate()[0]["dice"]
    assert per_case == pytest.approx(whole_split, rel=1e-9)


BATCHES = [1, 2, 5, 7, 8, 12]


def test_the_two_poolings_are_different_things_and_the_difference_needs_a_short_batch():
    """The arithmetic on its own, with no model in it - which is the only honest place for it.

    This control was written twice inside the trained test above and was flaky both times.
    Per batch size it demanded that every size not dividing 12 expose the old pooling, and
    `batch_size` 7 falsified that: whether two groupings of the same numbers average alike
    depends on the numbers. Taken once over all sizes it still failed about one run in three,
    because an unseeded one-epoch model can give every case the same score, and then no
    grouping of them differs from any other.

    **A control that depends on the luck of an initialisation is not a control.** The claim
    being guarded is about arithmetic, so it is asserted on numbers chosen here: a mean of
    group means is not the mean unless the groups are the same size.
    """
    scores = [0.1 * (i + 1) for i in range(12)]
    exact = float(np.mean(scores))
    for batch in BATCHES:
        chunks = [scores[i : i + batch] for i in range(0, len(scores), batch)]
        old = float(np.mean([np.mean(c) for c in chunks]))
        if 12 % batch:
            assert abs(old - exact) > 1e-9, (
                f"batch {batch} leaves a short last batch, so the two poolings must differ "
                "on numbers that are not all equal"
            )
        else:
            assert old == pytest.approx(exact, rel=1e-12), (
                f"batch {batch} divides 12, so the two must agree exactly"
            )


def test_the_split_score_is_the_mean_over_cases_whatever_the_batch_size(patients):
    """Because `batch_size` is a training setting and has no business in a reported score.

    One trained model, scored through loaders of different batch sizes, so the only variable
    is the batching, and five of the six sizes do not divide the 12 samples. No control for
    the old pooling here - that is the test above, where it does not depend on what a random
    initialisation happened to produce.
    """
    engine = Engine(spec_for(patients), device="cpu")
    engine.fit()
    run = engine.runs["tiny"]
    dataset = engine.dataset(run.spec, "validation")
    cases = engine._case_names(run.spec, "validation")
    expected = float(np.mean([row["dice"] for row in engine.evaluate_cases("tiny")]))

    for batch in BATCHES:
        reported = run.trainer.evaluate(make_loader(dataset, batch_size=batch), cases=cases)
        assert reported["val_dice"] == pytest.approx(expected, rel=1e-9), (
            f"at batch_size {batch} the reported figure is not the mean over cases"
        )


@pytest.mark.parametrize("batch", [1, 5, 7, 12])
def test_an_epoch_line_is_the_mean_over_examples_whatever_the_batching(patients, batch):
    """The half of `#196` the test above cannot see, and a mutation batch found that out.

    `evaluate()` passes case names, so its accumulating metrics are read off the per-case
    accumulator - which means the example weighting underneath is invisible there, and three
    mutations of it survived the first batch. It is not invisible on an **epoch line**: a
    training pass is shuffled, so there is no case identity to pool by and the division is
    all there is. Same for `loss` and `foreground` in any pass, which have no per-case
    definition at all.

    So this scores with no case names, which is the shape a training epoch has, and asserts
    that nothing moves with the batching.
    """
    engine = Engine(spec_for(patients), device="cpu")
    engine.fit()
    run = engine.runs["tiny"]
    dataset = engine.dataset(run.spec, "validation")

    reference = run.trainer.evaluate(make_loader(dataset, batch_size=3))
    got = run.trainer.evaluate(make_loader(dataset, batch_size=batch))

    # 12 samples, so 3 divides and the reference is the mean over examples by construction.
    for key in ("val_loss", "val_foreground", "val_dice"):
        assert got[key] == pytest.approx(reference[key], rel=1e-6), (
            f"{key} moved from {reference[key]:.8f} to {got[key]:.8f} between batch sizes "
            "3 and " + str(batch) + ", so a training setting is still in the number"
        )


def test_the_worst_case_is_one_of_the_cases_the_mean_was_taken_over(patients):
    """Two readings of one set of scores, which is what the epoch now computes once.

    Not arithmetic for its own sake: if the mean came from the accumulator and the minimum
    from somewhere else, an epoch could report a worst case that is not among the cases its
    mean covers, and `val_dice_worst` is a number callbacks act on.
    """
    engine = Engine(spec_for(patients), device="cpu")
    history = engine.fit()["tiny"]
    rows = engine.evaluate_cases("tiny")
    per_case = sorted(row["dice"] for row in rows)

    last = history.records[-1]
    assert last["val_dice_worst"] == pytest.approx(per_case[0], rel=1e-6)
    assert per_case[0] <= last["val_dice"] <= per_case[-1], (
        "the epoch's mean falls outside the range of the case scores its own worst came from"
    )


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
