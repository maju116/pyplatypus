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
from pyplatypus.objectives.metrics import Dice
from tests.conftest import write_png


@pytest.fixture
def patients(tmp_path):
    """Four patients, three slices each, with a square to find."""
    root = tmp_path / "data"
    for patient in range(4):
        for index in range(3):
            key = f"patient{patient:02d}_slice{index:03d}"
            mask = np.zeros((32, 32, 3), np.uint8)
            mask[6 + patient:18 + patient, 8:24] = 255
            write_png(root / key / "images" / "a.png", mask // 2)
            write_png(root / key / "masks" / "a.png", mask)
    return root


def spec_for(root, **model):
    block = {"name": "tiny", "input_shape": [32, 32], "n_class": 2, "blocks": 2,
             "filters": 4, "batch_size": 2, "epochs": 1,
             "metrics": [{"name": "dice"}]}
    block.update(model)
    return from_dict({
        "task": "semantic_segmentation",
        "data": {"train_path": str(root), "validation_path": str(root),
                 "colormap": [[0, 0, 0], [255, 255, 255]], "shuffle": False},
        "models": [block],
    })


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
        grid_prediction = torch.cat([torch.cat([hard[0], hard[1]], dim=1),
                                     torch.cat([hard[2], hard[3]], dim=1)], dim=0)
        grid_target = torch.cat([torch.cat([batch_y[0], batch_y[1]], dim=2),
                                 torch.cat([batch_y[2], batch_y[3]], dim=2)], dim=1)
        whole = metric(
            torch.nn.functional.one_hot(grid_prediction, 2).permute(2, 0, 1)[None].float(),
            grid_target[None],
        ).item()

        assert reported[sample.key] == pytest.approx(whole, abs=1e-5)


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
    summary = summarise_cases([
        {"group": "patient01", "dice": 0.4},
        {"group": "patient02", "dice": 0.8},
    ])[0]
    assert summary["worst_group"] == "patient01"


def test_summarising_nothing_is_an_error():
    with pytest.raises(EngineError, match="no case scores"):
        summarise_cases([])


def test_rows_without_metrics_are_an_error():
    with pytest.raises(EngineError, match="no metric columns"):
        summarise_cases([{"case": "a"}])
