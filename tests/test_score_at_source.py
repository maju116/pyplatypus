"""Scoring on the grid the files are on, which is the only one two models share.

`evaluate()` scores each model through its own loader, so a model at 16 is measured
against a truth reduced to 16 while a model at 64 is measured against the truth as stored.
Both are a real Dice of a real prediction against a real mask and they answer different
questions - measured on retinal vessels, the difference reported a resized model at +0.0187
where the honest figure was +0.0033. `maju116/rplatypus#187`.

The fixture is that measurement in miniature: a bar four pixels wide in a 64-pixel picture,
which is one pixel at 16 - thin enough that a reduced truth keeps it only just, which is
the regime the retinal measurement was in.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyplatypus import Engine, from_dict
from pyplatypus.engine import EngineError
from pyplatypus.objectives.metrics import Dice
from tests.conftest import write_png

BAR = (slice(20, 24), slice(8, 56))


@pytest.fixture
def thin(tmp_path):
    """Four cases at 64 x 64, each with a four-pixel bar to find."""
    root = tmp_path / "data"
    for case in range(4):
        mask = np.zeros((64, 64, 3), np.uint8)
        mask[BAR[0], BAR[1]] = 255
        write_png(root / f"case{case:02d}" / "images" / "a.png", mask // 2 + 20)
        write_png(root / f"case{case:02d}" / "masks" / "a.png", mask)
    return root


def spec_for(root, size):
    return from_dict(
        {
            "task": "semantic_segmentation",
            "data": {
                "train_path": str(root),
                "validation_path": str(root),
                "colormap": [[0, 0, 0], [255, 255, 255]],
                "shuffle": False,
            },
            "models": [
                {
                    "name": f"at{size}",
                    "input_shape": [size, size],
                    "blocks": 2,
                    "filters": 4,
                    "batch_size": 2,
                    "epochs": 1,
                    "metrics": [{"name": "dice"}],
                }
            ],
        }
    )


def truth_as_the_fixture_drew_it() -> torch.Tensor:
    """The target built from the fixture's own expression, not from the package's reader.

    A control that shares no code with what it checks: if `target_as_stored` ever read the
    mask at the model's size, or coded the colours the other way round, this would not
    follow it.
    """
    classes = np.zeros((64, 64), np.int64)
    classes[BAR[0], BAR[1]] = 1
    onehot = np.eye(2, dtype=np.float32)[classes]
    return torch.from_numpy(np.moveaxis(onehot, -1, 0))[None]


def test_the_source_figure_is_the_arithmetic_done_by_hand(thin):
    """The engine's number against the same number computed outside it, to seven places."""
    engine = Engine(spec_for(thin, 16), device="cpu")
    engine.fit()

    rows = {row["case"]: row["dice"] for row in engine.evaluate_cases("at16", at="source")}

    metric = Dice(smooth=1.0)
    truth = truth_as_the_fixture_drew_it()
    for case, probabilities in engine.predict_stream("at16", split="validation", space="source"):
        assert probabilities.shape == (64, 64, 2), "the prediction is not on the source grid"
        prediction = torch.from_numpy(np.moveaxis(probabilities, -1, 0).copy())[None]
        hard = torch.nn.functional.one_hot(prediction.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            expected = float(metric(hard, truth))
        assert rows[case] == pytest.approx(expected, abs=1e-7)


def test_the_two_grids_disagree_which_is_the_whole_point(thin):
    engine = Engine(spec_for(thin, 16), device="cpu")
    engine.fit()

    on_model = engine.evaluate(at="model")[0]
    on_source = engine.evaluate(at="source")[0]

    assert abs(on_model["dice"] - on_source["dice"]) > 1e-6
    # The column has always meant the grid this row's numbers were computed on.
    assert on_model["load_shape"] == "16 x 16"
    assert on_source["load_shape"] == "source"
    # A loss on a resampled prediction is a number with no training meaning.
    assert "loss" in on_model
    assert "loss" not in on_source


def test_a_model_already_at_the_source_grid_agrees_with_itself(thin):
    """The control: with nothing to resample, the two grids must give one number."""
    engine = Engine(spec_for(thin, 64), device="cpu")
    engine.fit()

    assert engine.evaluate(at="source")[0]["dice"] == pytest.approx(
        engine.evaluate(at="model")[0]["dice"], abs=1e-6
    )


def test_the_split_figure_is_the_mean_of_its_cases(thin):
    """`at` changes the grid and nothing else, so the pooling is what `at="model"` uses."""
    engine = Engine(spec_for(thin, 16), device="cpu")
    engine.fit()

    cases = [row["dice"] for row in engine.evaluate_cases("at16", at="source")]
    assert len(cases) == 4
    assert engine.evaluate(at="source")[0]["dice"] == pytest.approx(
        sum(cases) / len(cases), abs=1e-7
    )


def test_grouping_works_on_the_source_grid_too(thin):
    engine = Engine(spec_for(thin, 16), device="cpu")
    engine.fit()

    rows = engine.evaluate_cases("at16", group_by=r"^(case\d)\d$", at="source")
    assert {row["group"] for row in rows} == {"case0"}
    assert "case" not in rows[0]


def test_an_unknown_grid_is_refused_by_name(thin):
    engine = Engine(spec_for(thin, 16), device="cpu")
    engine.fit()

    with pytest.raises(EngineError, match="at is 'model' or 'source'"):
        engine.evaluate(at="native")
    with pytest.raises(EngineError, match="at is 'model' or 'source'"):
        engine.evaluate_cases("at16", at="native")


def test_the_warning_is_silent_when_the_rows_are_comparable(thin, recwarn):
    """Two models on different grids warn at `at="model"` and must not at `at="source"`."""
    spec = from_dict(
        {
            "task": "semantic_segmentation",
            "data": {
                "train_path": str(thin),
                "validation_path": str(thin),
                "colormap": [[0, 0, 0], [255, 255, 255]],
                "shuffle": False,
            },
            "models": [
                {
                    "name": name,
                    "input_shape": [size, size],
                    "blocks": 2,
                    "filters": 4,
                    "batch_size": 2,
                    "epochs": 1,
                    "metrics": [{"name": "dice"}],
                }
                for name, size in (("small", 16), ("big", 64))
            ],
        }
    )
    engine = Engine(spec, device="cpu")
    engine.fit()

    with pytest.warns(UserWarning, match="different grids"):
        engine.evaluate(at="model")

    recwarn.clear()
    rows = engine.evaluate(at="source")
    assert [row["load_shape"] for row in rows] == ["source", "source"]
    assert not [w for w in recwarn if "different grids" in str(w.message)]
