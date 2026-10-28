"""One label per image, from a directory to an answer.

The engine is mostly plumbing and the tests that matter are the ones about decisions: that
the split cut out of a training folder is stratified, that a validation score is not measured
on augmented images, that the table says which split it is about, and that the names come
from `classes` in order.
"""

from __future__ import annotations

import numpy as np
import pydantic
import pytest

from pyplatypus import ClassificationEngine, build_engine
from pyplatypus.data.splits import split_samples
from pyplatypus.engine import EngineError
from pyplatypus.errors import ConfigError
from pyplatypus.spec import ClassificationData, ClassificationModel, from_dict
from pyplatypus.spec.spec import ClassificationSpec
from tests.conftest import write_png


def picture(seed: int, bright: bool, size: int = 24) -> np.ndarray:
    generator = np.random.default_rng(seed)
    base = 200 if bright else 40
    return np.clip(generator.normal(base, 25, (size, size, 3)), 0, 255).astype(np.uint8)


def tree(root, counts: dict[str, int], size: int = 24):
    """One directory per class. The first class is dark and the rest are bright, so there
    is something to learn and a run that trains is distinguishable from one that does not.
    """
    seed = 0
    for index, (name, how_many) in enumerate(counts.items()):
        for number in range(how_many):
            write_png(root / name / f"{number:04d}.png", picture(seed, index > 0, size))
            seed += 1
    return root


def data(root, classes=("benign", "malignant"), **over) -> ClassificationData:
    """No `validation` here: the spec refuses `validation: false` beside `split`, because
    one says there is no validation set and the other says where it comes from."""
    return ClassificationData(**{"train_path": str(root), "classes": list(classes), **over})


def model(**over) -> ClassificationModel:
    return ClassificationModel(
        **{
            "name": "ours",
            "input_shape": (16, 16),
            "blocks": 2,
            "filters": 4,
            "epochs": 2,
            "batch_size": 8,
            "optimizer": {"name": "adam", "learning_rate": 0.01},
            **over,
        }
    )


def split(fractions=(0.6, 0.2, 0.2), group_by=None, seed=0) -> dict:
    return {"fractions": list(fractions), "group_by": group_by, "seed": seed}


def spec(block: ClassificationData, models: list[ClassificationModel], **over):
    """A whole specification, which is what the engine takes now that the tag exists."""
    return ClassificationSpec(**{"task": "classification", "data": block, "models": models, **over})


def engine_for(block, models, **over) -> ClassificationEngine:
    """`seed` and `output_dir` are the specification's fields, not the engine's, which is
    what changed when the tag landed: the engine takes one object and reads them off it."""
    return ClassificationEngine(spec(block, models, **over))


# --------------------------------------------------------------- the split, which is why


def test_the_split_is_stratified_where_an_unstratified_one_loses_a_class(tmp_path):
    """The decision this engine exists to get right, with the control beside it.

    The same samples through `split_samples` lose the rare class from a part; through the
    engine every part holds it. Both halves are asserted, because "every part has it" is
    worth nothing unless the alternative provably does not - measured over 400 seeds, a
    random 70/15/15 of 100 images at 3% prevalence leaves some part empty 91% of the time,
    and every per-class metric then refuses at the end of the first epoch.
    """
    root = tree(tmp_path / "set", {"benign": 97, "malignant": 3})
    block = data(root, split=split((0.7, 0.15, 0.15), seed=0))
    engine = engine_for(block, [model()], seed=0)

    for part in ("train", "validation", "test"):
        labels = {sample.label for sample in engine._samples[part]}
        assert labels == {"benign", "malignant"}, (part, labels)

    # the control: the unstratified split of the same discovery
    found = engine._discover(root, block).samples
    plain = split_samples(found, fractions=(0.7, 0.15, 0.15), group_by=None, seed=0)
    empty = [
        name
        for name in ("train", "validation", "test")
        if "malignant" not in {s.label for s in getattr(plain, name)}
    ]
    assert empty, "the unstratified split keeps every class here, so this test proves nothing"


def test_a_class_with_fewer_examples_than_parts_is_refused_by_name(tmp_path):
    root = tree(tmp_path / "set", {"benign": 40, "malignant": 2})
    with pytest.raises(ConfigError, match="class 'malignant' cannot be split"):
        engine_for(data(root, split=split((0.7, 0.15, 0.15))), [model()], seed=0)


def test_three_paths_are_read_without_a_split(tmp_path):
    for part, counts in (
        ("train", {"benign": 6, "malignant": 6}),
        ("validation", {"benign": 3, "malignant": 3}),
        ("test", {"benign": 2, "malignant": 2}),
    ):
        tree(tmp_path / part, counts)
    block = ClassificationData(
        train_path=str(tmp_path / "train"),
        validation_path=str(tmp_path / "validation"),
        test_path=str(tmp_path / "test"),
        classes=["benign", "malignant"],
    )
    engine = engine_for(block, [model()], seed=0)
    assert {k: len(v) for k, v in engine._samples.items()} == {
        "train": 12,
        "validation": 6,
        "test": 4,
    }
    assert engine.labelled == {"train", "validation", "test"}


# ------------------------------------------------------------------------ the data


def test_the_balance_is_reported_per_split(tmp_path):
    root = tree(tmp_path / "set", {"benign": 30, "malignant": 10})
    engine = engine_for(data(root, split=split()), [model()], seed=0)
    whole = sum(engine.balance(part).total for part in ("train", "validation", "test"))
    assert whole == 40
    assert engine.balance("train").counts["malignant"] >= 1
    # 10 of 40 is 25%, so the sentence about threshold metrics does not belong here - which
    # this asserted until the sentence became conditional, on a set it was never true of.
    assert "rarest: malignant" in str(engine.balance("train"))
    assert "below about 10%" not in str(engine.balance("train"))


def test_the_data_can_be_looked_at_before_anything_is_fitted(tmp_path):
    """Which is how somebody meets their own dataset, and is why `_model_spec` is not
    `_resolve_model`: that one answers out of the runs, and there are none yet."""
    root = tree(tmp_path / "set", {"benign": 8, "malignant": 8})
    engine = engine_for(data(root, split=split()), [model()], seed=0)
    assert len(engine.dataset(None, "train")) > 0
    assert engine.balance("train").total > 0
    with pytest.raises(EngineError, match="nothing has been trained"):
        engine.evaluate("validation")


def test_only_the_training_split_is_augmented(tmp_path):
    """A validation score measured on transformed images is a score for a dataset nobody
    has."""
    root = tree(tmp_path / "set", {"benign": 8, "malignant": 8})
    engine = engine_for(
        data(root, split=split()),
        [model(augmentation=[{"name": "HorizontalFlip", "params": {"p": 1.0}}])],
        seed=0,
    )
    assert engine.dataset(None, "train").augmenter is not None
    assert engine.dataset(None, "validation").augmenter is None
    assert engine.dataset(None, "test").augmenter is None
    # and it can be asked for anyway, which is what the argument is for
    assert engine.dataset(None, "validation", augment=True).augmenter is not None


def test_the_loader_shuffles_training_only_and_drops_a_lone_last_batch(tmp_path):
    root = tree(tmp_path / "set", {"benign": 20, "malignant": 20})
    engine = engine_for(data(root, split=split()), [model(batch_size=8)], seed=0)
    train = engine.loader(None, "train")
    validation = engine.loader(None, "validation")
    assert train.drop_last is True
    assert validation.drop_last is False
    assert train.batch_size == 8


# ------------------------------------------------------------------------- a run


def test_a_run_trains_scores_and_answers(tmp_path):
    """The join, the way a user reaches it: pictures in directories, two models, a table and
    a per-image answer with a key that can be joined to anything else."""
    root = tree(tmp_path / "set", {"benign": 24, "malignant": 24})
    engine = engine_for(
        data(root, split=split()),
        [model(name="ours", epochs=3), model(name="wider", filters=8, epochs=3)],
        output_dir=str(tmp_path / "out"),
        seed=0,
    )
    histories = engine.fit()
    assert sorted(histories) == ["ours", "wider"]
    assert all(len(history) == 3 for history in histories.values())

    rows = engine.evaluate("validation")
    assert [row["model"] for row in rows] == ["ours", "wider"]
    row = rows[0]
    assert row["split"] == "validation"
    assert row["architecture"] == "plain_cnn"
    assert row["loss_function"] == "cce"
    assert row["parameters"] > 0
    assert row["epochs_run"] == 3
    # bare metric names, because the `split` column is what says which pass this is
    assert "auc" in row and "val_auc" not in row
    assert "loss" in row and "val_loss" not in row

    answers = engine.answers(split="test")
    keys = [answer["key"] for answer in answers]
    assert keys == [sample.key for sample in engine._samples["test"]]
    assert len(set(keys)) == len(keys)
    for answer in answers:
        assert answer["predicted"] in ("benign", "malignant")
        assert 0.0 <= answer["probability"] <= 1.0
        assert answer["p_benign"] + answer["p_malignant"] == pytest.approx(1.0, abs=1e-5)

    probabilities = engine.predict("ours", "test")
    assert probabilities.shape == (len(engine._samples["test"]), 2)
    assert engine.predict_labels("ours", "test") == [a["predicted"] for a in answers]

    written = engine.write_history()
    assert sorted(path.name for path in written.values()) == [
        "ours_history.csv",
        "wider_history.csv",
    ]
    assert "epoch,train_loss" in written["ours"].read_text()


def test_the_specifications_seed_is_what_makes_a_run_reproducible(tmp_path):
    """`spec.seed` rather than an engine argument, which is what moved when the tag landed -
    and nothing tested it until a mutation that stopped reading it survived.

    It is a different seed from `split.seed`: that one decides which images go where, this
    one decides the initial weights and the shuffle. So the test holds the split fixed and
    varies only this, which is the only way to tell the two apart.
    """
    import torch

    from pyplatypus.models.classifier import build_classifier

    root = tree(tmp_path / "set", {"benign": 16, "malignant": 16})
    block = data(root, split=split(seed=0))

    def initial_weights(seed):
        """The weights the seed provably decides, read **before** training.

        The first draft of this compared the weights after an epoch and failed on two runs
        of the same seed: training on CUDA is not bit-reproducible without
        `deterministic: true`, so what it was measuring was the kernels and not the seed.
        Construction is the thing `seed_everything` reaches.
        """
        engine = engine_for(block, [model(epochs=1)], seed=seed)
        network = build_classifier(engine.spec.models[0], n_class=engine.spec.data.n_class)
        return torch.cat(
            [
                value.flatten()
                for value in network.state_dict().values()
                if value.is_floating_point()
            ]
        )

    same = [initial_weights(0), initial_weights(0)]
    other = initial_weights(7)
    assert torch.equal(same[0], same[1]), "the same seed gave two different networks"
    assert not torch.equal(same[0], other), "two seeds gave the same network"


def test_the_default_for_best_model_is_the_threshold_free_metric(tmp_path):
    """And a metric nobody asked for is refused with what is available, rather than with a
    KeyError out of the table."""
    root = tree(tmp_path / "set", {"benign": 16, "malignant": 16})
    engine = engine_for(data(root, split=split()), [model(metrics=[{"name": "auc"}])], seed=0)
    engine.fit()
    assert engine.best_model() == "ours"
    assert engine.best_model("auc") == "ours"
    with pytest.raises(EngineError, match="no model reports 'accuracy'"):
        engine.best_model("accuracy")


# --------------------------------------------------------------------- refusals


def test_a_run_needs_at_least_one_model_and_unique_names(tmp_path):
    """**The specification refuses both, and the engine does not repeat it.** It carried its
    own guards while it took its pieces loose; `PlatypusSpec.unique_model_names` and
    `min_length=1` were already there, so keeping them would have been a second copy of one
    refusal - checked rather than assumed before the engine's were removed.
    """
    root = tree(tmp_path / "set", {"benign": 8, "malignant": 8})
    block = data(root, split=split())
    with pytest.raises(pydantic.ValidationError, match="at least 1 item"):
        engine_for(block, [])
    with pytest.raises(pydantic.ValidationError, match="names must be unique"):
        engine_for(block, [model(name="a"), model(name="a")])


def test_the_engine_refuses_another_task_by_name_and_says_which_one_trains_it(tmp_path):
    """`build_engine` picks by the tag, so this is reached by somebody holding the wrong
    engine - and the message names the right one rather than saying "not classification"."""
    from pyplatypus.spec.spec import SegmentationSpec

    other = SegmentationSpec(
        task="semantic_segmentation",
        data={
            "train_path": str(tmp_path),
            "validation": False,
            "colormap": [[0, 0, 0], [255, 255, 255]],
        },
        models=[
            {
                "name": "m",
                "architecture": "u_net",
                "input_shape": [16, 16],
                "blocks": 1,
                "filters": 2,
            }
        ],
    )
    with pytest.raises(EngineError, match="`Engine` trains that one"):
        ClassificationEngine(other)


def test_build_engine_returns_this_engine_for_the_tag(tmp_path):
    """The whole point of the tag: holding a specification is enough."""
    root = tree(tmp_path / "set", {"benign": 8, "malignant": 8})
    built = build_engine(spec(data(root, split=split()), [model()]))
    assert isinstance(built, ClassificationEngine)


def test_a_specification_round_trips_through_from_dict(tmp_path):
    """Which is the thing that was impossible until the tag landed, and is what makes every
    slice before this one reachable from a file."""
    root = tree(tmp_path / "set", {"benign": 8, "malignant": 8})
    loaded = from_dict(
        {
            "task": "classification",
            "data": {
                "train_path": str(root),
                "classes": ["benign", "malignant"],
                "split": split(),
            },
            "models": [
                {"name": "ours", "input_shape": [16, 16], "blocks": 2, "filters": 4, "epochs": 1}
            ],
        }
    )
    assert isinstance(loaded, ClassificationSpec)
    assert loaded.task.value == "classification"
    assert [m.name for m in loaded.models] == ["ours"]
    assert [m.name for m in loaded.models[0].metrics] == ["auc"]
    engine = build_engine(loaded)
    assert isinstance(engine, ClassificationEngine)
    assert engine.balance("train").total > 0


def test_a_model_nobody_named_is_refused_with_the_names_there_are(tmp_path):
    root = tree(tmp_path / "set", {"benign": 8, "malignant": 8})
    engine = engine_for(data(root, split=split()), [model(name="ours")], seed=0)
    with pytest.raises(EngineError, match="no model named 'theirs'"):
        engine.dataset("theirs", "train")


def test_a_split_this_run_does_not_have_is_refused(tmp_path):
    root = tree(tmp_path / "set", {"benign": 8, "malignant": 8})
    engine = engine_for(data(root, split=split((0.8, 0.2))), [model()], seed=0)
    assert "test" not in engine._samples
    with pytest.raises(EngineError):
        engine.dataset(None, "test")


# ------------------------------------------------- a label that is a set, end to end


FINDINGS = ["atelectasis", "cardiomegaly", "effusion"]


def films(root, how_many: int = 30, size: int = 24) -> str:
    """A CSV with one column per finding, and pictures whose brightness carries the first.

    Each finding is present on a fixed arithmetic subset, so every one of them has both
    sides in every part of a 0.6/0.2/0.2 split and nothing here depends on a draw. The
    brightness tracks `atelectasis`, so a run that learns is distinguishable from one that
    does not on at least one class.
    """
    root.mkdir(parents=True, exist_ok=True)
    header = "key,images,Atelectasis,Cardiomegaly,Effusion"
    lines = [header]
    for index in range(how_many):
        present = (index % 2 == 0, index % 3 == 0, index % 4 == 0)
        write_png(root / f"{index:04d}.png", picture(index, present[0], size))
        cells = ",".join("1" if yes else "0" for yes in present)
        lines.append(f"film{index:04d},{index:04d}.png,{cells}")
    path = root / "films.csv"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def multi_data(csv, **over) -> ClassificationData:
    return ClassificationData(
        **{
            "train_path": csv,
            "mode": "config_file",
            "classes": list(FINDINGS),
            "labels_from": ["Atelectasis", "Cardiomegaly", "Effusion"],
            **over,
        }
    )


def test_the_columns_reach_the_reader_from_the_specification_alone(tmp_path):
    """The one line that decides what a configured run reads. The control is the same CSV
    with `labels_from` left out, which is then a file with no `label` column at all."""
    csv = films(tmp_path / "chest")
    engine = engine_for(multi_data(csv, split=split()), [model()])

    assert engine.spec.data.multilabel is True
    assert all(sample.labels is not None for sample in engine._samples["train"])

    with pytest.raises(ConfigError, match="needs a 'label' column"):
        engine_for(data(csv, classes=FINDINGS, mode="config_file", split=split()), [model()])


def test_the_balance_of_a_split_reports_shares_that_do_not_add_up(tmp_path):
    csv = films(tmp_path / "chest")
    balance = engine_for(multi_data(csv, split=split()), [model()]).balance("train")

    assert balance.labels_are_a_set
    assert balance.total == 18, "0.6 of 30 films"
    assert balance.labels_per_image > 1.0
    assert sum(balance.counts.values()) > balance.total
    assert "do not add up" in str(balance)
    assert balance.never == [] and balance.always == []


def test_every_finding_reaches_every_part_of_the_split(tmp_path):
    """Which is the reason the split dispatches rather than striating, and the control is
    the same films through `split_samples`, where it does not hold."""
    csv = films(tmp_path / "chest")
    engine = engine_for(multi_data(csv, split=split()), [model()])

    for part in ("train", "validation", "test"):
        members = engine._samples[part]
        for finding in FINDINGS:
            positive = sum(1 for sample in members if finding in sample.labels)
            assert 0 < positive < len(members), f"{finding} in {part}"


def test_a_run_over_a_set_of_labels_trains_scores_and_answers(tmp_path):
    """The join this slice is for: a configuration, the reader, the stratified split, the
    multi-hot target, the metric family and the two ways of asking what the model said."""
    csv = films(tmp_path / "chest")
    engine = engine_for(
        multi_data(csv, split=split()), [model(metrics=[{"name": "auc"}, {"name": "f1"}])]
    )
    histories = engine.fit()

    assert engine.runs["ours"].trainer.multilabel is True
    logs = histories["ours"].records[-1]
    assert {"train_loss", "val_loss", "train_auc", "val_auc", "train_f1"} <= set(logs)

    rows = engine.evaluate("test")
    assert len(rows) == 1
    assert rows[0]["split"] == "test"
    assert 0.0 <= rows[0]["auc"] <= 1.0

    probabilities = engine.predict("ours", "test")
    assert probabilities.shape == (6, 3)
    assert not np.allclose(probabilities.sum(axis=1), 1.0), (
        "independent questions: a softmax here would make them sum to one"
    )

    named = engine.predict_labels("ours", "test")
    assert all(isinstance(answer, list) for answer in named)
    assert all(set(answer) <= set(FINDINGS) for answer in named)


def test_answers_carries_the_truth_and_the_prediction_as_lists_and_no_probability(tmp_path):
    """`probability` is the chosen class's own number and with independent questions nothing
    was chosen, so the column is absent rather than invented. The `p_<class>` columns are
    each class's answer and they are the whole of it."""
    csv = films(tmp_path / "chest")
    engine = engine_for(multi_data(csv, split=split()), [model(epochs=1)])
    engine.fit()
    rows = engine.answers("ours", "test")

    assert len(rows) == 6
    first = rows[0]
    assert "probability" not in first
    assert "label" not in first, "the truth is a set here and is called `labels`"
    assert isinstance(first["labels"], list) and isinstance(first["predicted"], list)
    assert set(first) == {"key", "labels", "predicted"} | {f"p_{name}" for name in FINDINGS}

    # The truth comes back in `classes` order whatever order the columns were read in, and
    # it is the CSV's own answer for that film rather than anything the model said.
    by_key = {row["key"]: row for row in rows}
    for sample in engine._samples["test"]:
        assert by_key[sample.key]["labels"] == [name for name in FINDINGS if name in sample.labels]
    # And `predicted` agrees with the probabilities beside it, at 0.5.
    for row in rows:
        assert row["predicted"] == [name for name in FINDINGS if row[f"p_{name}"] > 0.5]


def test_the_single_label_row_is_unchanged(tmp_path):
    """The control on the branch in `answers`: one label out of alternatives still carries
    `label`, one `predicted` name and its `probability`."""
    root = tree(tmp_path / "train", {"benign": 12, "malignant": 12})
    engine = engine_for(data(root, split=split()), [model(epochs=1)])
    engine.fit()
    row = engine.answers("ours", "test")[0]

    assert isinstance(row["predicted"], str)
    assert "probability" in row and "labels" not in row
    assert row["label"] in ("benign", "malignant")
