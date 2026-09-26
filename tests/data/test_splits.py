"""Splitting, and the several ways a split can be quietly wrong.

The interesting tests here are not that the fractions come out roughly right. They are
that a patient cannot appear on both sides, that the same inputs give the same split on
any machine, and that a `group_by` which matches nothing is refused instead of silently
becoming one-patient-per-slice - the bug that makes a bad result look good.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pyplatypus.data.paths import Sample, discover_samples
from pyplatypus.data.splits import (
    Split,
    group_of,
    split_dataset,
    split_samples,
    write_splits,
)
from pyplatypus.errors import ConfigError
from tests.conftest import write_png


def sample(key: str) -> Sample:
    return Sample(key=key, images=(f"/data/{key}.png",), masks=(f"/masks/{key}.png",))


def slices(patients: int, per_patient: int) -> list[Sample]:
    return [
        sample(f"patient{p:02d}_slice{s:03d}")
        for p in range(patients)
        for s in range(per_patient)
    ]


PATIENT = r"^(patient\d+)_"


# --------------------------------------------------------------------- grouping
def test_without_a_pattern_every_sample_is_its_own_group():
    assert group_of("case_17", None) == "case_17"


def test_the_first_capture_group_names_the_patient():
    assert group_of("patient03_slice120", PATIENT) == "patient03"


def test_a_pattern_without_a_capture_group_uses_the_whole_match():
    assert group_of("patient03_slice120", r"patient\d+") == "patient03"


def test_a_pattern_that_matches_nothing_is_an_error_not_a_fallback():
    # The whole reason this is an error: falling back to one group per sample would put
    # slices of one patient on both sides of the split and report a better score for it.
    with pytest.raises(ConfigError, match="does not match"):
        group_of("case_17", PATIENT)


# ---------------------------------------------------------------------- splitting
def test_no_patient_appears_in_two_splits():
    split = split_samples(slices(10, 5), group_by=PATIENT, seed=1)
    assert split.groups_of("train").isdisjoint(split.groups_of("validation"))
    assert split.groups_of("train").isdisjoint(split.groups_of("test"))
    assert split.groups_of("validation").isdisjoint(split.groups_of("test"))


def test_every_sample_lands_exactly_once():
    samples = slices(10, 5)
    split = split_samples(samples, group_by=PATIENT)
    landed = [s.key for name in ("train", "validation", "test") for s in split[name]]
    assert sorted(landed) == sorted(s.key for s in samples)
    assert len(landed) == len(set(landed)) == 50


def test_the_same_seed_gives_the_same_split():
    first = split_samples(slices(12, 3), group_by=PATIENT, seed=7)
    again = split_samples(slices(12, 3), group_by=PATIENT, seed=7)
    assert first.groups_of("validation") == again.groups_of("validation")


def test_a_different_seed_gives_a_different_split():
    a = split_samples(slices(12, 3), group_by=PATIENT, seed=1)
    b = split_samples(slices(12, 3), group_by=PATIENT, seed=2)
    assert a.groups_of("validation") != b.groups_of("validation")


def test_sample_order_does_not_change_the_split():
    # Groups are assigned in a sorted, then seeded-shuffled order, so a dataset listed in
    # a different order splits the same way. Without this, rerunning on a machine whose
    # directory listing differs would silently produce a different experiment.
    samples = slices(9, 4)
    forwards = split_samples(samples, group_by=PATIENT, seed=3)
    backwards = split_samples(list(reversed(samples)), group_by=PATIENT, seed=3)
    assert forwards.groups_of("train") == backwards.groups_of("train")


def test_fractions_are_respected_in_samples_not_groups():
    # 40 patients, 5 slices each: 70/15/15 of 200 samples.
    split = split_samples(slices(40, 5), fractions=(0.7, 0.15, 0.15), group_by=PATIENT)
    assert split.counts["train"] == pytest.approx(140, abs=10)
    assert split.counts["validation"] == pytest.approx(30, abs=10)
    assert split.counts["test"] == pytest.approx(30, abs=10)


def test_uneven_groups_still_land_near_the_target_share():
    # Patients with wildly different numbers of slices, which is the normal case. Assigning
    # by group count rather than sample count would miss the target badly here.
    samples = [
        sample(f"patient{p:02d}_slice{s:03d}")
        for p in range(20)
        for s in range(1 + (p % 7) * 3)
    ]
    split = split_samples(samples, fractions=(0.6, 0.4), group_by=PATIENT, seed=5)
    share = split.counts["train"] / len(samples)
    assert 0.5 < share < 0.7


def test_two_fractions_means_no_test_split():
    split = split_samples(slices(8, 2), fractions=(0.8, 0.2), group_by=PATIENT)
    assert split.counts["test"] == 0
    assert split.counts["train"] + split.counts["validation"] == 16


def test_a_mapping_of_fractions_works_too():
    split = split_samples(
        slices(10, 2), fractions={"train": 0.5, "validation": 0.5}, group_by=PATIENT
    )
    assert split.counts["test"] == 0


def test_one_lopsided_patient_does_not_starve_a_split():
    # One patient with 100 slices and two with one each. Chasing target sizes alone leaves
    # a split empty here; being a little off the fractions beats returning an empty
    # validation or test set.
    samples = (
        [sample(f"patient00_slice{s:03d}") for s in range(100)]
        + [sample("patient01_slice000"), sample("patient02_slice000")]
    )
    split = split_samples(samples, group_by=PATIENT, seed=0)
    assert all(split.counts[name] > 0 for name in ("train", "validation", "test"))


def test_one_sample_per_patient_needs_no_pattern():
    split = split_samples([sample(f"case{i:02d}") for i in range(10)])
    assert sum(split.counts.values()) == 10
    assert all(split.counts[name] > 0 for name in ("train", "validation", "test"))


# ------------------------------------------------------------------- refusals
def test_too_few_groups_for_the_splits_asked_for():
    with pytest.raises(ConfigError, match="cannot fill"):
        split_samples(slices(2, 30), group_by=PATIENT)


def test_fractions_have_to_add_up():
    with pytest.raises(ConfigError, match="add up to 1"):
        split_samples(slices(5, 2), fractions=(0.5, 0.2), group_by=PATIENT)


def test_validation_cannot_be_given_nothing():
    with pytest.raises(ConfigError, match="validation"):
        split_samples(slices(5, 2), fractions={"train": 1.0}, group_by=PATIENT)


def test_negative_fractions_are_refused():
    with pytest.raises(ConfigError, match="negative"):
        split_samples(slices(5, 2), fractions=(1.2, -0.2), group_by=PATIENT)


def test_an_unknown_split_name_is_refused():
    with pytest.raises(ConfigError, match="unknown split"):
        split_samples(slices(5, 2), fractions={"train": 0.5, "holdout": 0.5})


def test_nothing_to_split():
    with pytest.raises(ConfigError, match="no samples"):
        split_samples([])


# ---------------------------------------------------------------------- writing
def test_written_csvs_are_readable_by_the_config_file_mode(tmp_path):
    root = tmp_path / "dataset"
    for patient in range(6):
        for index in range(3):
            key = f"patient{patient:02d}_slice{index:03d}"
            write_png(root / key / "images" / "a.png", np.zeros((8, 8, 3)))
            write_png(root / key / "masks" / "a.png", np.zeros((8, 8, 3)))

    report = split_dataset(root, tmp_path / "splits", group_by=PATIENT, seed=0)

    assert set(report["paths"]) == {"train", "validation", "test"}
    assert sum(report["samples"].values()) == 18
    assert sum(report["groups"].values()) == 6

    # The point of writing CSVs at all: they are what a spec can point at.
    found = discover_samples(report["paths"]["train"], mode="config_file")
    assert len(found) == report["samples"]["train"]
    assert all(path.exists() for sample in found.samples for path in sample.images)


def test_the_csv_records_which_patient_each_row_came_from(tmp_path):
    root = tmp_path / "dataset"
    for patient in range(4):
        for index in range(2):
            key = f"patient{patient:02d}_slice{index:03d}"
            write_png(root / key / "images" / "a.png", np.zeros((8, 8, 3)))
            write_png(root / key / "masks" / "a.png", np.zeros((8, 8, 3)))

    report = split_dataset(root, tmp_path / "splits", group_by=PATIENT, fractions=(0.5, 0.5))

    seen = {}
    for name in ("train", "validation"):
        with open(report["paths"][name], newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert {"key", "group", "images", "masks"} <= set(rows[0])
        seen[name] = {row["group"] for row in rows}
    # Provenance is not decoration: this is how anyone checks the split afterwards.
    assert seen["train"].isdisjoint(seen["validation"])


def test_paths_are_written_relative_so_the_dataset_can_move(tmp_path):
    root = tmp_path / "dataset"
    write_png(root / "patient00_slice000" / "images" / "a.png", np.zeros((8, 8, 3)))
    write_png(root / "patient00_slice000" / "masks" / "a.png", np.zeros((8, 8, 3)))
    write_png(root / "patient01_slice000" / "images" / "a.png", np.zeros((8, 8, 3)))
    write_png(root / "patient01_slice000" / "masks" / "a.png", np.zeros((8, 8, 3)))

    split = split_samples(
        discover_samples(root).samples, fractions=(0.5, 0.5), group_by=PATIENT
    )
    written = write_splits(split, tmp_path, relative=True)
    with open(written["train"], newline="") as handle:
        row = next(csv.DictReader(handle))
    # Path(), not a leading slash: on Windows an absolute path starts with a drive letter,
    # and asserting on '/' tested the platform rather than the behaviour.
    assert not Path(row["images"]).is_absolute()

    written = write_splits(split, tmp_path / "elsewhere", relative=False)
    with open(written["train"], newline="") as handle:
        row = next(csv.DictReader(handle))
    assert Path(row["images"]).is_absolute()


def test_an_empty_split_is_not_written_as_an_empty_file(tmp_path):
    root = tmp_path / "dataset"
    for patient in range(4):
        key = f"patient{patient:02d}_slice000"
        write_png(root / key / "images" / "a.png", np.zeros((8, 8, 3)))
        write_png(root / key / "masks" / "a.png", np.zeros((8, 8, 3)))

    report = split_dataset(root, tmp_path / "splits", fractions=(0.75, 0.25),
                           group_by=PATIENT)
    assert "test" not in report["paths"]
    assert not (tmp_path / "splits" / "test.csv").exists()


def test_split_is_indexable_and_complains_about_unknown_names():
    split = split_samples(slices(6, 2), group_by=PATIENT)
    assert isinstance(split, Split)
    assert len(split["train"]) > 0
    with pytest.raises(KeyError, match="no split called"):
        split["holdout"]


def test_a_split_can_be_trained_on_without_touching_anything_else(tmp_path):
    """The claim that matters: what split_dataset() writes is what a spec can train from.

    Discovery working is not enough - the CSVs have to survive the whole pipeline, or
    splitting would be a tool that produces files nothing else accepts.
    """
    from pyplatypus import Engine, from_dict

    root = tmp_path / "dataset"
    for patient in range(6):
        for index in range(2):
            key = f"patient{patient:02d}_slice{index:03d}"
            mask = np.zeros((32, 32, 3), np.uint8)
            mask[8:24, 8:24] = 255
            write_png(root / key / "images" / "a.png", mask // 2)
            write_png(root / key / "masks" / "a.png", mask)

    report = split_dataset(root, tmp_path / "splits", group_by=PATIENT,
                           fractions=(0.5, 0.5), seed=0)

    engine = Engine(from_dict({
        "data": {
            "train_path": report["paths"]["train"],
            "validation_path": report["paths"]["validation"],
            "mode": "config_file",
            "colormap": [[0, 0, 0], [255, 255, 255]],
        },
        "models": [{"name": "tiny", "input_shape": [32, 32], "n_class": 2, "blocks": 2,
                    "filters": 4, "batch_size": 2, "epochs": 1,
                    "metrics": [{"name": "dice"}]}],
    }), device="cpu")
    history = engine.fit()["tiny"]
    assert len(history) == 1
    assert history.records[0]["val_dice"] > 0
