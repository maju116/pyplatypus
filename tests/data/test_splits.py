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
    SPLIT_NAMES,
    Split,
    group_of,
    split_dataset,
    split_samples,
    stratified_split,
    write_splits,
)
from pyplatypus.errors import ConfigError
from tests.conftest import write_png


def sample(key: str) -> Sample:
    return Sample(key=key, images=(f"/data/{key}.png",), masks=(f"/masks/{key}.png",))


def slices(patients: int, per_patient: int) -> list[Sample]:
    return [
        sample(f"patient{p:02d}_slice{s:03d}") for p in range(patients) for s in range(per_patient)
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
        sample(f"patient{p:02d}_slice{s:03d}") for p in range(20) for s in range(1 + (p % 7) * 3)
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
    samples = [sample(f"patient00_slice{s:03d}") for s in range(100)] + [
        sample("patient01_slice000"),
        sample("patient02_slice000"),
    ]
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

    split = split_samples(discover_samples(root).samples, fractions=(0.5, 0.5), group_by=PATIENT)
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

    report = split_dataset(root, tmp_path / "splits", fractions=(0.75, 0.25), group_by=PATIENT)
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

    report = split_dataset(
        root, tmp_path / "splits", group_by=PATIENT, fractions=(0.5, 0.5), seed=0
    )

    engine = Engine(
        from_dict(
            {
                "task": "semantic_segmentation",
                "data": {
                    "train_path": report["paths"]["train"],
                    "validation_path": report["paths"]["validation"],
                    "mode": "config_file",
                    "colormap": [[0, 0, 0], [255, 255, 255]],
                },
                "models": [
                    {
                        "name": "tiny",
                        "input_shape": [32, 32],
                        "blocks": 2,
                        "filters": 4,
                        "batch_size": 2,
                        "epochs": 1,
                        "metrics": [{"name": "dice"}],
                    }
                ],
            }
        ),
        device="cpu",
    )
    history = engine.fit()["tiny"]
    assert len(history) == 1
    assert history.records[0]["val_dice"] > 0


# ---------------------------------------------------------------- the stratified split


def labelled(counts: dict[str, int]) -> list[Sample]:
    return [
        Sample(key=f"{label}/{number:04d}", images=(Path("x.png"),), label=label)
        for label, how_many in counts.items()
        for number in range(how_many)
    ]


def test_every_part_holds_every_class_where_a_random_split_would_not():
    """The claim, with the control beside it: the same samples through `split_samples` lose
    the rare class from a part. Measured over 400 seeds, a random 70/15/15 of 100 images at
    3% prevalence leaves some part empty 91% of the time, and every per-class metric then
    refuses to average over a class its pass never held."""
    samples = labelled({"common": 97, "rare": 3})
    strat = stratified_split(samples, fractions=(0.7, 0.15, 0.15), seed=0)
    for name in ("train", "validation", "test"):
        assert {s.label for s in getattr(strat, name)} == {"common", "rare"}, name

    plain = split_samples(samples, fractions=(0.7, 0.15, 0.15), seed=0)
    empty = [
        name
        for name in ("train", "validation", "test")
        if "rare" not in {s.label for s in getattr(plain, name)}
    ]
    assert empty, "the random split keeps every class on this draw, so this proves nothing"


def test_it_costs_at_most_an_image_of_the_fractions_asked_for():
    """What it is traded against, stated as a number rather than as a reassurance: the
    worst drift over 200 seeds at 100 images was 0.0100, which is one image."""
    samples = labelled({"common": 97, "rare": 3})
    worst = 0.0
    for seed in range(20):
        strat = stratified_split(samples, fractions=(0.7, 0.15, 0.15), seed=seed)
        for name, want in (("train", 0.7), ("validation", 0.15), ("test", 0.15)):
            worst = max(worst, abs(len(getattr(strat, name)) / 100 - want))
    assert worst <= 0.02, worst


def test_a_stratified_split_is_deterministic_in_its_seed():
    """Named for this function rather than `test_the_same_seed_gives_the_same_split`, which
    this file already has for `split_samples` - the first draft shadowed it, so the original
    would have stopped running while still appearing to. Caught by `ruff`'s F811, which is
    the instrument for exactly that and is why the lint gate is a gate."""
    samples = labelled({"common": 40, "rare": 10})
    first = stratified_split(samples, fractions=(0.6, 0.2, 0.2), seed=3)
    again = stratified_split(samples, fractions=(0.6, 0.2, 0.2), seed=3)
    other = stratified_split(samples, fractions=(0.6, 0.2, 0.2), seed=4)

    def keys(part):
        return [sample.key for sample in part]

    assert keys(first.train) == keys(again.train)
    assert keys(first.train) != keys(other.train)


def test_a_group_holding_two_labels_is_refused_rather_than_cut_in_half():
    """Stratifying divides by label first, so such a group is handed out once per label it
    contains and lands on both sides - which is the leak `group_by` exists to prevent.

    Measured on six patients with two of them holding both labels: `patient06` appeared in
    train and in validation, where the same data split without stratifying leaked nothing.
    So the combination is refused rather than silently preferring one guarantee.
    """
    samples = []
    for patient in range(1, 7):
        labels = ["common", "rare"] if patient in (5, 6) else ["common", "common"]
        for index, label in enumerate(labels):
            samples.append(
                Sample(
                    key=f"patient{patient:02d}_img{index}",
                    images=(Path("x.png"),),
                    label=label,
                )
            )
    pattern = r"^(patient\d+)_"
    with pytest.raises(ConfigError, match="more than one label"):
        stratified_split(samples, fractions=(0.5, 0.5), group_by=pattern)

    # the control: the same data splits fine without stratifying, and keeps patients whole
    plain = split_samples(samples, fractions=(0.5, 0.5), group_by=pattern)
    on_both_sides = {group_of(s.key, pattern) for s in plain.train} & {
        group_of(s.key, pattern) for s in plain.validation
    }
    assert on_both_sides == set()


def test_a_group_of_one_class_each_is_allowed_and_stays_whole():
    """The other half: `group_by` is only refused when a group genuinely spans labels."""
    samples = []
    for patient in range(1, 9):
        label = "rare" if patient > 5 else "common"
        for index in range(2):
            samples.append(
                Sample(
                    key=f"patient{patient:02d}_img{index}",
                    images=(Path("x.png"),),
                    label=label,
                )
            )
    pattern = r"^(patient\d+)_"
    strat = stratified_split(samples, fractions=(0.5, 0.5), group_by=pattern)

    def groups(part):
        return {group_of(sample.key, pattern) for sample in part}

    assert groups(strat.train) & groups(strat.validation) == set()
    assert {s.label for s in strat.train} == {"common", "rare"}
    assert {s.label for s in strat.validation} == {"common", "rare"}


def test_a_sample_with_no_label_is_refused_by_name():
    samples = labelled({"common": 10, "rare": 10})
    samples.append(Sample(key="unlabelled/1", images=(Path("x.png"),)))
    with pytest.raises(ConfigError, match="carry no label"):
        stratified_split(samples, fractions=(0.5, 0.5))


def test_a_class_with_fewer_examples_than_parts_is_refused_naming_the_class():
    """`split_samples`' own refusal talks about groups and patients, which is right for the
    caller it was written for and wrong here - so it is wrapped in one that says `class`."""
    with pytest.raises(ConfigError, match="class 'rare' cannot be split"):
        stratified_split(labelled({"common": 50, "rare": 2}), fractions=(0.7, 0.15, 0.15))
    # two parts and two examples is enough, which is the boundary rather than a number
    assert stratified_split(labelled({"common": 50, "rare": 2}), fractions=(0.5, 0.5))


def test_an_empty_set_is_refused():
    with pytest.raises(ConfigError, match="no samples to split"):
        stratified_split([], fractions=(0.5, 0.5))


# ----------------------------------------------- the split for a label that is a set


FINDINGS = ["atelectasis", "cardiomegaly", "effusion", "nodule"]
PREVALENCE = [0.10, 0.05, 0.03, 0.02]


def films(n: int, seed: int = 0, *, group_size: int = 1) -> list[Sample]:
    """`n` films, four independent findings at a chest film's prevalences.

    `group_size` puts several films under one patient key, so the same draw can be split
    with and without `group_by` and the two answers compared.
    """
    rng = np.random.default_rng(seed)
    present = rng.random((n, len(FINDINGS))) < np.array(PREVALENCE)
    return [
        Sample(
            key=f"patient{index // group_size:05d}_film{index:05d}",
            images=(Path(f"{index:05d}.png"),),
            labels=tuple(name for name, yes in zip(FINDINGS, row) if yes),
        )
        for index, row in enumerate(present)
    ]


def one_sided(split, classes=FINDINGS) -> list[tuple[str, str]]:
    """The (part, class) pairs where a part holds only one side of a class.

    That is the state every per-class metric for a set of labels refuses to average over,
    and it is what the split exists to prevent - so the test asserts on the pairs rather
    than on a count, because the pair says which one.
    """
    bad = []
    for name in SPLIT_NAMES:
        members = split[name]
        if not members:
            continue
        for klass in classes:
            positive = sum(1 for s in members if klass in s.labels)
            if positive == 0 or positive == len(members):
                bad.append((name, klass))
    return bad


def test_a_set_of_labels_gives_every_class_to_every_part():
    """And the control is the same draw split without stratifying, which does not.

    Without it this asserts only that one arrangement of 500 films happens to be fine -
    which the first draft of this test did, at 1000 films where a random split is fine
    87.75% of the time. Measured over 400 draws: random leaves some part one-sided 51.25%
    of the time at 500 films and 12.25% at 1000, where this is 0.00% at both. So the size
    is chosen to make the control *likely* rather than to make the claim bigger.
    """
    samples = films(500, seed=0)
    assert one_sided(stratified_split(samples, fractions=(0.7, 0.15, 0.15), seed=0)) == []

    plain = split_samples(samples, fractions=(0.7, 0.15, 0.15), seed=0)
    assert one_sided(plain) == [("validation", "nodule")], (
        "the control: a random split of this draw leaves validation with no nodule at all"
    )


def test_it_holds_at_every_seed_rather_than_at_a_lucky_one():
    for seed in range(25):
        strat = stratified_split(films(600, seed=seed), fractions=(0.7, 0.15, 0.15), seed=seed)
        assert one_sided(strat) == [], f"seed {seed}"


def test_the_fractions_are_not_what_it_costs():
    """The size deficit is the second key, so the parts come out where they were asked for.
    0.0000 was the worst drift over 400 draws; this is the one-draw version of that."""
    samples = films(1000, seed=3)
    strat = stratified_split(samples, fractions=(0.7, 0.15, 0.15), seed=3)
    for name, share in zip(SPLIT_NAMES, (0.7, 0.15, 0.15)):
        assert len(strat[name]) / len(samples) == pytest.approx(share, abs=0.01)


def test_a_class_too_rare_for_the_smallest_part_is_refused_in_those_numbers():
    """Four positives and a 15% part is 0.6 of an image, so no assignment can give the
    smallest part one. The refusal says that rather than returning a split that dies at the
    end of the first epoch - and the same data at 50/50 is fine, which is the boundary."""

    def finding(index: int) -> tuple[str, ...]:
        # Written out rather than drawn: four is the whole point, and `common` carries
        # enough of both sides that the refusal can only be about the other one.
        return ("rare",) * (index < 4) + ("common",) * (index % 2 == 0 and index < 50)

    samples = [
        Sample(key=f"film{index:04d}", images=(Path("x.png"),), labels=finding(index))
        for index in range(100)
    ]
    with pytest.raises(ConfigError, match="'rare'"):
        stratified_split(samples, fractions=(0.7, 0.15, 0.15))
    # Two parts and four positives is enough, which is the boundary rather than a number.
    halved = stratified_split(samples, fractions=(0.5, 0.5))
    assert one_sided(halved, ["rare", "common"]) == []


def test_a_patient_stays_whole_and_every_class_still_reaches_every_part():
    """The group is the unit of assignment here, which is how both guarantees hold at once.

    The single-label path cannot do this - it refuses a group spanning two labels, because
    its strata would hand that group out twice - and a group spanning several findings is
    the ordinary case for a set.
    """
    samples = films(900, seed=11, group_size=3)
    strat = stratified_split(samples, fractions=(0.7, 0.15, 0.15), group_by=PATIENT, seed=11)

    assert one_sided(strat) == []
    seen = [strat.groups_of(name) for name in SPLIT_NAMES]
    assert seen[0] & seen[1] == set() and seen[0] & seen[2] == set() and seen[1] & seen[2] == set()


def test_the_same_seed_is_the_same_split_and_another_seed_is_not():
    samples = films(400, seed=1)
    first = stratified_split(samples, fractions=(0.6, 0.2, 0.2), seed=3)
    again = stratified_split(samples, fractions=(0.6, 0.2, 0.2), seed=3)
    other = stratified_split(samples, fractions=(0.6, 0.2, 0.2), seed=4)

    assert [s.key for s in first.train] == [s.key for s in again.train]
    assert [s.key for s in first.train] != [s.key for s in other.train]


def test_a_dataset_labelled_two_ways_at_once_is_refused_here_too():
    """The two need different algorithms, so there is no split that is right for both
    halves - which is the same refusal the dataset makes, one layer down."""
    samples = films(50, seed=0)
    samples.append(Sample(key="odd", images=(Path("x.png"),), label="atelectasis"))
    with pytest.raises(ConfigError, match="one label where the rest carry a set"):
        stratified_split(samples, fractions=(0.5, 0.5))


def test_a_rare_class_inside_a_common_one_is_what_the_order_is_for():
    """The rarest class is placed first, and reversing that passed every other test here.

    What it changes is not whether a returned split is correct - the post-condition
    guarantees that - but whether a split that exists is found at all. Measured over 200
    draws of a rare finding occurring only inside a common one: 3.00% refused at 500 films
    against 35.50% with the order reversed, and 0.00% against 12.00% at 1000. On the ten
    seeds below, rarest-first finds a split every time and commonest-first refuses on five
    of them.

    This is the clinically ordinary shape rather than a constructed one: an effusion inside
    a film already marked abnormal.
    """
    films = [
        Sample(
            key=f"film{index:05d}",
            images=(Path("x.png"),),
            # Written out rather than drawn, so the fixture does not depend on a random
            # stream staying the same across a numpy upgrade. Ten rare films, every one of
            # them inside a common one.
            labels=("common",) * (index % 2 == 0) + ("rare",) * (index % 66 == 0),
        )
        for index in range(600)
    ]
    assert sum(1 for f in films if "rare" in f.labels) == 10

    for seed in range(10):
        strat = stratified_split(films, fractions=(0.7, 0.15, 0.15), seed=seed)
        assert one_sided(strat, ["common", "rare"]) == [], f"seed {seed}"
