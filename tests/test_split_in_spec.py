"""The split the configuration asked for, actually performed.

The spec tests say the block is accepted; these say it divides, that it divides the way
`group_by` was told to, and that an empty split is refused rather than trained on.
"""

import numpy as np
import pytest
from PIL import Image

from pyplatypus.engine import Engine
from pyplatypus.spec import from_dict

COLORMAP = [[0, 0, 0], [255, 255, 255]]


@pytest.fixture
def four_patients(tmp_path):
    """Twelve samples, three per patient - so splitting by file and by patient differ."""
    root = tmp_path / "all"
    for patient in range(4):
        for slice_no in range(3):
            sample = root / f"patient{patient}_slice{slice_no}"
            (sample / "images").mkdir(parents=True)
            (sample / "masks").mkdir(parents=True)
            mask = np.zeros((32, 32), np.uint8)
            mask[8:20, 8:20] = 255
            Image.fromarray(mask).convert("RGB").save(sample / "images" / "a.png")
            Image.fromarray(mask).convert("RGB").save(sample / "masks" / "m.png")
    return root


def engine_for(root, split):
    return Engine(from_dict({
        "task": "semantic_segmentation",
        "data": {"train_path": str(root), "colormap": COLORMAP, "split": split},
        "models": [{"name": "m", "input_shape": [32, 32]}],
    }), device="cpu", check_masks=False)


def patients_in(engine, split):
    return {sample.key.split("_")[0] for sample in engine._samples[split]}


def test_two_fractions_give_train_and_validation(four_patients):
    engine = engine_for(four_patients, {"fractions": [0.75, 0.25], "group_by": None})
    assert set(engine._samples) == {"train", "validation"}
    assert len(engine._samples["train"]) + len(engine._samples["validation"]) == 12


def test_three_fractions_give_a_test_set_too(four_patients):
    engine = engine_for(four_patients, {"fractions": [0.5, 0.25, 0.25], "group_by": None})
    assert set(engine._samples) == {"train", "validation", "test"}
    assert sum(len(v) for v in engine._samples.values()) == 12


def test_group_by_keeps_a_patient_out_of_both_halves(four_patients):
    """The reason the field is required, demonstrated rather than described.

    Without it the same patient appears on both sides, and validation then measures memory
    rather than generalisation - several points of Dice too high, with nothing in the
    output to say so.
    """
    by_file = engine_for(four_patients, {"fractions": [0.5, 0.25, 0.25], "group_by": None})
    by_patient = engine_for(four_patients,
                            {"fractions": [0.5, 0.25, 0.25], "group_by": r"^(patient\d+)_"})

    assert patients_in(by_file, "train") & patients_in(by_file, "validation")
    assert not (patients_in(by_patient, "train") & patients_in(by_patient, "validation"))
    assert not (patients_in(by_patient, "train") & patients_in(by_patient, "test"))


def test_the_split_is_deterministic(four_patients):
    split = {"fractions": [0.5, 0.25, 0.25], "group_by": r"^(patient\d+)_", "seed": 7}
    first = engine_for(four_patients, split)
    again = engine_for(four_patients, split)
    assert ([s.key for s in first._samples["validation"]]
            == [s.key for s in again._samples["validation"]])


def test_a_different_seed_divides_differently(four_patients):
    """Otherwise the seed is decoration and nobody could re-split a stubborn dataset."""
    keys = []
    for seed in (1, 2, 3, 4, 5):
        engine = engine_for(four_patients, {"fractions": [0.5, 0.25, 0.25],
                                            "group_by": r"^(patient\d+)_", "seed": seed})
        keys.append(tuple(s.key for s in engine._samples["validation"]))
    assert len(set(keys)) > 1


def test_fewer_groups_than_splits_is_refused_through_the_spec(tmp_path):
    """The refusal belongs to `split_samples`; this says the engine does not swallow it.

    There is deliberately no emptiness check in the engine. Measured before deciding: 12
    samples at (0.999, 0.001) give validation one sample rather than none, because
    `split_samples` guarantees every split it is asked for gets at least one - so a guard
    here could never have fired, and a check that cannot fire reads as protection while
    being none.
    """
    root = tmp_path / "two"
    for patient in range(2):
        for slice_no in range(3):
            sample = root / f"patient{patient}_slice{slice_no}"
            (sample / "images").mkdir(parents=True)
            (sample / "masks").mkdir(parents=True)
            mask = np.zeros((32, 32), np.uint8)
            mask[8:20, 8:20] = 255
            Image.fromarray(mask).convert("RGB").save(sample / "images" / "a.png")
            Image.fromarray(mask).convert("RGB").save(sample / "masks" / "m.png")

    with pytest.raises(Exception, match="cannot fill"):
        engine_for(root, {"fractions": [0.4, 0.3, 0.3], "group_by": r"^(patient\d+)_"})


def test_a_tiny_share_still_gets_a_sample_rather_than_nothing(four_patients):
    engine = engine_for(four_patients, {"fractions": [0.999, 0.001], "group_by": None})
    assert len(engine._samples["validation"]) >= 1


def test_nothing_is_written_to_disk(four_patients, tmp_path):
    """A specification is a description. `split_dataset()` is the tool when the CSVs are
    the point; being read is not supposed to leave files behind."""
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    engine_for(four_patients, {"fractions": [0.5, 0.25, 0.25], "group_by": None})
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == before


def test_a_test_set_cut_from_training_data_is_scoreable(four_patients):
    """What the two changes mean together, and neither says on its own.

    A separate `test_path` may be images alone, so it is recorded as unlabelled and
    `evaluate` refuses it. A test split cut out of the *training* folder cannot be: it came
    from data that has masks, by construction. So this one is scoreable, and saying so is
    the difference between a third fraction being useful and being a trap.
    """
    engine = engine_for(four_patients, {"fractions": [0.5, 0.25, 0.25], "group_by": None})
    assert "test" in engine.labelled

    _, mask = engine.dataset(engine.spec.models[0], "test")[0]
    assert mask is not None

    engine.fit()
    row = engine.evaluate("test")[0]
    scored = [m.name for m in engine.spec.models[0].metrics]
    assert scored and all(row[name] is not None for name in scored)
