"""A label that is a set: one column per class, and what stops being true when it is.

The claims here are the reader's second shape. `pyplatypus#307`'s slice 2: that the columns
are the classes paired by position and not by header, that a cell which is neither a 0 nor
a 1 is refused rather than mapped, that an image with none of the findings is a labelled
image, that the shares stop adding up to one, and that a dataset labelled two ways at once
is refused rather than resolved.

**No `labels_from` field appears anywhere in here**, and that is the slice rather than an
omission: the configuration surface lands last, after the trainer, so that nothing in a
YAML file can describe a feature a run would get wrong. The columns are passed to the
reader, which is the one place that honours them either way.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pyplatypus.data.classification import (
    ClassBalance,
    ClassificationDataset,
    LabelError,
    discover_labelled,
)
from pyplatypus.errors import ConfigError
from pyplatypus.models import build_classifier
from pyplatypus.objectives.classification import build_classification_metric
from pyplatypus.spec import ClassificationData, ClassificationModel
from pyplatypus.spec.components import AucMetric
from pyplatypus.training.torch_data import make_classification_loader
from tests.conftest import write_png

FINDINGS = ["atelectasis", "cardiomegaly", "effusion", "nodule"]
COLUMNS = ["Atelectasis", "Cardiomegaly", "Effusion", "Nodule"]


def picture(seed: int, size: int = 24) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def data(classes=FINDINGS, **over) -> ClassificationData:
    return ClassificationData(
        **{
            "train_path": ".",
            "validation": False,
            "mode": "config_file",
            "classes": list(classes),
            **over,
        }
    )


def model(**over) -> ClassificationModel:
    return ClassificationModel(**{"name": "stage", "input_shape": (16, 16), **over})


def films(tmp_path, rows: list[tuple[int, int, int, int]], *, header=None, extra="") -> str:
    """A CSV with one column per finding, and the pictures it names.

    `rows` holds the cells as written, so a test can put `-1` or nothing in one.
    """
    root = tmp_path / "chest"
    root.mkdir(exist_ok=True)
    for number in range(len(rows)):
        write_png(root / f"{number:04d}.png", picture(number))
    names = header if header is not None else COLUMNS
    lines = ["key,images," + ",".join(names) + extra]
    for number, cells in enumerate(rows):
        lines.append(f"film{number:04d},{number:04d}.png," + ",".join(str(cell) for cell in cells))
    path = root / "films.csv"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# --------------------------------------------------------------- the columns are the classes


def test_one_column_per_class_becomes_a_set_of_names_in_class_order(tmp_path):
    csv = films(tmp_path, [(1, 0, 1, 0), (0, 0, 0, 0), (0, 1, 0, 1)])
    found = discover_labelled(csv, data(), label_columns=COLUMNS)

    assert [sample.labels for sample in found.samples] == [
        ("atelectasis", "effusion"),
        (),
        ("cardiomegaly", "nodule"),
    ]
    # The names are the data's `classes`, not the CSV's headers: the column says where to
    # read and the class list says what it is called.
    assert all(sample.label is None for sample in found.samples)


def test_an_image_with_none_of_the_findings_is_labelled_and_not_skipped(tmp_path):
    csv = films(tmp_path, [(0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 0, 0)])
    found = discover_labelled(csv, data(), label_columns=COLUMNS)

    assert len(found.samples) == 3, "an empty set is an answer; `None` is the absence of one"
    assert found.samples[0].labels == ()
    assert found.skipped == ()


def test_the_position_is_the_class_index_and_not_the_alphabet(tmp_path):
    """The headers are given in one order and the classes in another: the pairing is by
    position, which is what `channels_from` and `classes_from` both promise."""
    csv = films(tmp_path, [(1, 0), (0, 1)], header=["Second", "First"])
    found = discover_labelled(
        csv, data(classes=["first", "second"]), label_columns=["Second", "First"]
    )

    assert [sample.labels for sample in found.samples] == [("first",), ("second",)]


def test_a_cell_that_is_neither_a_zero_nor_a_one_is_refused_naming_it(tmp_path):
    csv = films(tmp_path, [(1, 0, 0, 0), (0, -1, 0, 0)])
    with pytest.raises(ConfigError, match=r"film0001.*'-1'.*'Cardiomegaly'"):
        discover_labelled(csv, data(), label_columns=COLUMNS)


def test_an_empty_cell_is_not_a_zero(tmp_path):
    csv = films(tmp_path, [(1, 0, 0, 0), (0, "", 0, 0)])
    with pytest.raises(ConfigError, match=r"film0001.*'Cardiomegaly' empty"):
        discover_labelled(csv, data(), label_columns=COLUMNS)


def test_a_float_is_read_because_that_is_what_a_csv_gets_written_as(tmp_path):
    csv = films(
        tmp_path,
        [("1.0", "0.0", "1.0", "0.0"), ("0.0", "1.0", "0.0", "1.0"), (0, 0, 0, 0)],
    )
    found = discover_labelled(csv, data(), label_columns=COLUMNS)
    assert found.samples[0].labels == ("atelectasis", "effusion")


def test_a_missing_column_is_refused_and_named(tmp_path):
    csv = films(tmp_path, [(1, 0, 0)], header=COLUMNS[:3])
    with pytest.raises(ConfigError, match="Nodule"):
        discover_labelled(csv, data(), label_columns=COLUMNS)


def test_one_column_per_class_means_as_many_columns_as_classes(tmp_path):
    csv = films(tmp_path, [(1, 0, 0, 0)])
    with pytest.raises(ConfigError, match="4 .*3 column"):
        discover_labelled(csv, data(), label_columns=COLUMNS[:3])


def test_a_set_of_labels_cannot_be_read_from_one_directory_per_class(tmp_path):
    root = tmp_path / "train"
    (root / "atelectasis").mkdir(parents=True)
    write_png(root / "atelectasis" / "0.png", picture(0))
    with pytest.raises(ConfigError, match="config_file"):
        discover_labelled(root, data(mode="nested_dirs"), label_columns=COLUMNS)


def test_the_label_column_is_not_required_when_the_columns_are_read(tmp_path):
    """A CSV carrying both is read as a set, and its `label` column is one of the many
    columns this reader ignores - the same way it ignores `group` or anything else.

    The control is the second call: with no columns given, the same file reads as one label
    per image, so neither route is quietly reading the other's cells.
    """
    two = ["atelectasis", "cardiomegaly"]
    csv = Path(films(tmp_path, [(1, 0), (0, 1)], header=COLUMNS[:2]))
    lines = csv.read_text().splitlines()
    rewritten = [lines[0] + ",label"] + [f"{line},{name}" for line, name in zip(lines[1:], two)]
    csv.write_text("\n".join(rewritten) + "\n")

    as_a_set = discover_labelled(csv, data(classes=two), label_columns=COLUMNS[:2])
    assert [sample.labels for sample in as_a_set.samples] == [
        ("atelectasis",),
        ("cardiomegaly",),
    ]
    assert all(sample.label is None for sample in as_a_set.samples)

    as_one_name = discover_labelled(csv, data(classes=two))
    assert [sample.label for sample in as_one_name.samples] == two
    assert all(sample.labels is None for sample in as_one_name.samples)


# ------------------------------------------------------- a class with only one side of it


def test_a_class_no_image_carries_is_refused_by_name(tmp_path):
    csv = films(tmp_path, [(1, 0, 1, 0), (0, 1, 1, 0), (1, 1, 0, 0)])
    with pytest.raises(LabelError, match="nodule"):
        discover_labelled(csv, data(), label_columns=COLUMNS)


def test_a_class_every_image_carries_is_refused_too(tmp_path):
    """The other direction, and it only exists for a set of labels: a finding present
    everywhere has no negative, so a specificity has nothing to average."""
    csv = films(tmp_path, [(1, 1, 0, 1), (1, 0, 1, 1), (1, 1, 1, 0)])
    with pytest.raises(LabelError, match="atelectasis"):
        discover_labelled(csv, data(), label_columns=COLUMNS)


# ------------------------------------------------------------------------------ the balance


def test_the_shares_are_each_class_own_and_the_total_is_images(tmp_path):
    csv = films(tmp_path, [(1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 1)])
    dataset = ClassificationDataset(
        discover_labelled(csv, data(), label_columns=COLUMNS).samples, model(), data()
    )
    balance = dataset.balance()

    assert balance.counts == {
        "atelectasis": 3,
        "cardiomegaly": 2,
        "effusion": 2,
        "nodule": 2,
    }
    assert balance.total == 4, "four films, nine labels"
    assert balance.labels_per_image == pytest.approx(9 / 4)
    assert balance.prevalence == pytest.approx(2 / 4)
    assert balance.labels_are_a_set
    assert "labels per image" in str(balance)
    assert "do not add up" in str(balance)


def test_one_label_out_of_alternatives_is_the_same_number_counted_either_way():
    """The control on `total`: where the counts do partition the images, giving the image
    count and deriving it from the sum have to agree, or the new field changed the old
    reading."""
    counts = {"benign": 97, "malignant": 3}
    derived = ClassBalance(counts=counts)
    given = ClassBalance(counts=counts, images=100)

    assert derived.total == given.total == 100
    assert derived.prevalence == given.prevalence == pytest.approx(0.03)
    assert derived.labels_per_image == given.labels_per_image == 1.0
    assert not derived.labels_are_a_set and given.labels_are_a_set


def test_a_class_with_only_one_side_of_it_is_named_in_the_report():
    one_sided = ClassBalance(counts={"a": 0, "b": 5, "c": 5}, images=5)
    assert one_sided.never == ["a"]
    assert one_sided.always == ["b", "c"]
    assert "never present" in str(one_sided)
    assert "present in every image" in str(one_sided)

    healthy = ClassBalance(counts={"a": 1, "b": 3}, images=5)
    assert healthy.never == [] and healthy.always == []
    assert "never present" not in str(healthy)


# ------------------------------------------------------------------------------ the dataset


def labelled(tmp_path, rows):
    spec = data()
    csv = films(tmp_path, rows)
    samples = discover_labelled(csv, spec, label_columns=COLUMNS).samples
    return ClassificationDataset(samples, model(), spec)


def test_an_item_is_the_image_and_a_multi_hot_row(tmp_path):
    dataset = labelled(tmp_path, [(1, 0, 1, 0), (0, 1, 0, 1)])
    image, target = dataset[0]

    assert dataset.multilabel
    assert image.shape == (16, 16, 3)
    assert target.dtype == np.float32, "the loss multiplies by it rather than indexing with it"
    assert target.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert dataset.labels().shape == (2, 4)


def test_a_dataset_labelled_two_ways_at_once_is_refused(tmp_path):
    """Neither reading is right for all of it and neither is visible in any number a run
    produces: read as exclusive, a film's second finding is lost to the argmax; read as a
    set, a film labelled with one name reaches a sigmoid head as all-zeros."""
    spec = data(classes=["atelectasis", "cardiomegaly"])
    csv = films(tmp_path, [(1, 0), (0, 1)], header=COLUMNS[:2])
    samples = list(discover_labelled(csv, spec, label_columns=COLUMNS[:2]).samples)
    mixed = (
        samples[0],
        replace(samples[1], label="cardiomegaly", labels=None),
    )

    with pytest.raises(LabelError, match="one label where the rest carry a set"):
        ClassificationDataset(mixed, model(), spec)


def test_the_loader_stacks_the_rows_into_examples_by_classes(tmp_path):
    dataset = labelled(tmp_path, [(1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1)])
    images, targets = next(iter(make_classification_loader(dataset, batch_size=4)))

    assert images.shape == (4, 3, 16, 16)
    assert targets.shape == (4, 4), "examples by classes, which is the logits' own shape"
    assert targets.dtype == torch.float32
    # Not transposed: row 0 is the first film's findings and column 0 is atelectasis
    # everywhere. A square batch is the shape where a transpose is invisible, so this is
    # asserted on one that is square on purpose.
    assert targets[0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert targets[:, 1].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_a_dataset_on_disk_reaches_a_head_and_a_multi_label_metric(tmp_path):
    """The join, for the reason the single-label file has one: green on each half proves
    the halves. Reader to loader to head to a metric that holds a table per class."""
    dataset = labelled(
        tmp_path, [(1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 0, 1)]
    )
    network = build_classifier(model(blocks=2, filters=4), n_class=4)
    metric = build_classification_metric(AucMetric(), multilabel=True)

    for images, targets in make_classification_loader(dataset, batch_size=2):
        with torch.no_grad():
            metric.update(network(images), targets)

    assert 0.0 <= metric.compute() <= 1.0
