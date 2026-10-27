"""One image, one label, and the two places that label can be written.

The claims here are the reader's: that a class index comes from the position in `classes`
and not from the directory listing, that an image with no label is refused rather than
guessed at, and that the prevalence is reported rather than left to be discovered from a
score. The last test is the join - reader, loader, head, metric - because green on each
half proves the halves.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyplatypus.data.classification import (
    ClassBalance,
    ClassificationDataset,
    LabelError,
    discover_labelled,
)
from pyplatypus.data.paths import discover_samples
from pyplatypus.errors import ConfigError
from pyplatypus.models import build_classifier
from pyplatypus.objectives.classification import Accuracy, Auc
from pyplatypus.spec import ClassificationData, ClassificationModel
from pyplatypus.training.torch_data import make_classification_loader
from tests.conftest import write_png


def picture(seed: int, size: int = 24) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def tree(root, counts: dict[str, int], size: int = 24):
    """One directory per class, `counts[name]` pictures inside each."""
    seed = 0
    for name, how_many in counts.items():
        for number in range(how_many):
            write_png(root / name / f"{number:04d}.png", picture(seed, size))
            seed += 1
    return root


def data(classes=("benign", "malignant"), **over) -> ClassificationData:
    return ClassificationData(
        **{"train_path": ".", "validation": False, "classes": list(classes), **over}
    )


def model(**over) -> ClassificationModel:
    return ClassificationModel(**{"name": "stage", "input_shape": (16, 16), **over})


# ------------------------------------------------------------------- one dir per class


def test_the_directory_is_the_label_and_the_key_names_both(tmp_path):
    root = tree(tmp_path / "train", {"benign": 2, "malignant": 3})
    found = discover_labelled(root, data())

    assert len(found) == 5
    assert [sample.label for sample in found.samples] == ["benign"] * 2 + ["malignant"] * 3
    # Numbered from zero inside each class, which is how these datasets actually arrive -
    # so the class in the key is doing work here rather than in a contrived case.
    assert [sample.key for sample in found.samples] == [
        "benign/0000.png",
        "benign/0001.png",
        "malignant/0000.png",
        "malignant/0001.png",
        "malignant/0002.png",
    ]


def test_two_classes_holding_the_same_filename_are_two_cases(tmp_path):
    """Which is why the key carries the class. A split writes the key, so a duplicate would
    make two different images one case - and the common layout is exactly this, both classes
    numbered from one."""
    root = tmp_path / "train"
    write_png(root / "benign" / "0001.png", picture(1))
    write_png(root / "malignant" / "0001.png", picture(2))
    keys = [sample.key for sample in discover_labelled(root, data()).samples]
    assert keys == ["benign/0001.png", "malignant/0001.png"]
    assert len(set(keys)) == 2


def test_a_file_at_the_top_has_no_label_and_is_refused(tmp_path):
    root = tree(tmp_path / "train", {"benign": 1, "malignant": 1})
    write_png(root / "stray.png", picture(9))
    with pytest.raises(ConfigError, match="has no label"):
        discover_labelled(root, data())


def test_a_directory_that_is_not_a_named_class_is_refused_or_skipped(tmp_path):
    """`strict_labels` is the whole of the difference, and the reason it defaults to true is
    that skipping an image removes it from the training set entirely, where skipping an
    object leaves the rest of its image in."""
    root = tree(tmp_path / "train", {"benign": 2, "malignant": 2, "unknown": 3})

    with pytest.raises(LabelError, match="'unknown' is a directory"):
        discover_labelled(root, data())

    loose = discover_labelled(root, data(strict_labels=False), strict=False)
    assert len(loose) == 4
    assert len(loose.skipped) == 3
    assert all("not a named class" in why for _, why in loose.skipped)


def test_a_class_with_no_images_is_refused_by_name(tmp_path):
    """Three layers earlier than the alternative: every per-class metric refuses to average
    over a class the pass never contained, and accuracy would report a figure that looks
    fine. `classes` is what to shorten when a class is genuinely not in a dataset."""
    root = tree(tmp_path / "train", {"benign": 2})
    with pytest.raises(LabelError, match="holds no image of"):
        discover_labelled(root, data())

    (root / "malignant").mkdir()
    with pytest.raises(LabelError, match="holds no image of"):
        discover_labelled(root, data())


# ------------------------------------------------------------------------- the CSV route


def write_csv(path, rows, header="images,label"):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def test_the_label_column_holds_a_name_and_the_images_resolve_against_the_csv(tmp_path):
    """The flag the shared reader gained, and the thing that flag must not break: a relative
    image path resolves against the CSV so that a config file travels with its data."""
    base = tmp_path / "set"
    write_png(base / "pics" / "a.png", picture(1))
    write_png(base / "pics" / "b.png", picture(2))
    csv = write_csv(base / "train.csv", ["pics/a.png,benign", "pics/b.png,malignant"])

    found = discover_labelled(csv, data(mode="config_file"))
    assert [sample.label for sample in found.samples] == ["benign", "malignant"]
    assert [sample.image for sample in found.samples] == [
        base / "pics" / "a.png",
        base / "pics" / "b.png",
    ]
    assert all(sample.masks == () for sample in found.samples)


def test_a_key_column_beats_the_row_number_on_this_route_too(tmp_path):
    base = tmp_path / "set"
    write_png(base / "a.png", picture(1))
    write_png(base / "b.png", picture(2))
    csv = write_csv(
        base / "train.csv",
        ["case-7,a.png,benign", "case-9,b.png,malignant"],
        header="key,images,label",
    )
    assert [s.key for s in discover_labelled(csv, data(mode="config_file")).samples] == [
        "case-7",
        "case-9",
    ]


def test_a_csv_may_name_anything_so_the_label_is_checked_against_classes(tmp_path):
    base = tmp_path / "set"
    for name in ("a", "b", "c"):
        write_png(base / f"{name}.png", picture(ord(name)))
    csv = write_csv(
        base / "train.csv", ["a.png,benign", "b.png,malignant", "c.png,benigh"]
    )  # a typo, which is the case this is for

    with pytest.raises(LabelError, match="benigh"):
        discover_labelled(csv, data(mode="config_file"))

    loose = discover_labelled(csv, data(mode="config_file", strict_labels=False), strict=False)
    assert len(loose) == 2
    assert any("benigh" in why for _, why in loose.skipped)


def test_a_row_with_no_label_is_incomplete_rather_than_unlabelled(tmp_path):
    base = tmp_path / "set"
    write_png(base / "a.png", picture(1))
    write_png(base / "b.png", picture(2))
    write_png(base / "c.png", picture(3))
    csv = write_csv(base / "train.csv", ["a.png,benign", "b.png,", "c.png,malignant"])
    with pytest.raises(ConfigError, match="label name"):
        discover_labelled(csv, data(mode="config_file"))


def test_the_shared_reader_still_reads_a_mask_column_as_paths(tmp_path):
    """The control for the flag: with it unset nothing about the segmentation route moved,
    and `label` is None - which is what lets one `Discovery` serve every task."""
    base = tmp_path / "set"
    write_png(base / "i.png", picture(1))
    write_png(base / "m.png", picture(2))
    csv = write_csv(base / "train.csv", ["i.png,m.png"], header="images,masks")
    found = discover_samples(csv, mode="config_file")
    assert found.samples[0].masks == (base / "m.png",)
    assert found.samples[0].label is None


def test_an_unlabelled_directory_of_pictures_is_not_this_layout(tmp_path):
    root = tree(tmp_path / "train", {"benign": 1, "malignant": 1})
    with pytest.raises(ConfigError, match="no unlabelled form"):
        discover_labelled(root, data(), only_images=True)


# ------------------------------------------------------------------- the index and the set


def test_the_class_index_is_the_position_in_classes_and_not_alphabetical(tmp_path):
    """The one thing in this reader that must not come off the disk.

    `classes` is written in reverse alphabetical order here on purpose: read off the
    directory listing the indices would come out the other way round, and a model trained
    where 0 meant `malignant` and used where it means `benign` reports plausible
    probabilities with nothing downstream able to tell.
    """
    root = tree(tmp_path / "train", {"benign": 1, "malignant": 1})
    spec = data(classes=("malignant", "benign"))
    found = discover_labelled(root, spec)
    dataset = ClassificationDataset(found.samples, model(), spec)

    assert spec.classes == ["malignant", "benign"]
    assert [sample.label for sample in found.samples] == ["benign", "malignant"]
    assert dataset.labels().tolist() == [1, 0]
    assert min(found.samples, key=lambda s: s.key).label == "benign"


def test_the_balance_is_reported_in_classes_order_and_names_the_rarest(tmp_path):
    root = tree(tmp_path / "train", {"benign": 97, "malignant": 3})
    spec = data()
    dataset = ClassificationDataset(discover_labelled(root, spec).samples, model(), spec)
    balance = dataset.balance()

    assert list(balance.counts) == ["benign", "malignant"]
    assert balance.counts == {"benign": 97, "malignant": 3}
    assert balance.total == 100
    assert balance.rarest == ("malignant", 3)
    assert balance.prevalence == pytest.approx(0.03)
    assert balance.to_dict()["prevalence"] == pytest.approx(0.03)
    assert "malignant" in str(balance) and "3.00%" in str(balance)
    assert "below about 10%" in str(balance), "the sentence belongs on a set this lopsided"


def test_the_balance_order_is_classes_and_not_arrival(tmp_path):
    """`classes` reversed against the directory listing, which is the only arrangement that
    can tell the two apart.

    The test above cannot: it names its classes in the order their directories are walked,
    so counting in arrival order passes it. Found by mutating the counting and watching
    nothing fail - §4's fixture that never contained the case, in a test written the same
    hour.
    """
    root = tree(tmp_path / "train", {"benign": 2, "malignant": 5})
    spec = data(classes=("malignant", "benign"))
    dataset = ClassificationDataset(discover_labelled(root, spec).samples, model(), spec)

    assert dataset.samples[0].label == "benign"
    assert list(dataset.balance().counts) == ["malignant", "benign"]


def test_a_set_with_nothing_usable_in_it_is_refused_rather_than_empty(tmp_path):
    """The tail every walk shares, reached where it is reachable: a CSV whose label column
    exported blank, with `strict=False` so the incomplete rows are dropped rather than
    refused - and then there is nothing left, which is a different message and the honest
    one. Also found by a surviving mutation."""
    base = tmp_path / "set"
    for name in ("a", "b"):
        write_png(base / f"{name}.png", picture(ord(name)))
    csv = write_csv(base / "train.csv", ["a.png,", "b.png,"])
    with pytest.raises(ConfigError, match="no usable samples"):
        discover_labelled(csv, data(mode="config_file"), strict=False)


def test_the_sentence_about_threshold_metrics_only_appears_where_it_is_true():
    """A warning that fires on a balanced set is a warning people learn to skip, and then it
    is not there on the set it was written for."""
    lopsided = ClassBalance(counts={"benign": 97, "malignant": 3})
    balanced = ClassBalance(counts={"benign": 12, "malignant": 12})
    assert "below about 10%" in str(lopsided)
    assert "below about 10%" not in str(balanced)
    # and the figure itself is reported either way, because it is a fact rather than advice
    assert "50.00%" in str(balanced)


def test_the_rarest_class_is_the_one_reported_whichever_it_is():
    """Three classes have no positive, so the share that predicts whether a threshold metric
    will say anything is the smallest one rather than the second class's."""
    balance = ClassBalance(counts={"a": 500, "b": 10, "c": 490})
    assert balance.rarest == ("b", 10)
    assert balance.prevalence == pytest.approx(0.01)


def test_an_item_is_the_image_at_input_shape_and_an_index(tmp_path):
    root = tree(tmp_path / "train", {"benign": 1, "malignant": 1}, size=40)
    spec = data()
    dataset = ClassificationDataset(discover_labelled(root, spec).samples, model(), spec)

    image, label = dataset[0]
    assert image.shape == (16, 16, 3)
    assert image.dtype == np.float32 and 0.0 <= image.min() and image.max() <= 1.0
    assert label == 0
    assert isinstance(label, np.int64)
    assert dataset[-1][1] == 1

    with pytest.raises(IndexError):
        dataset[2]


def test_only_images_hands_back_no_label(tmp_path):
    root = tree(tmp_path / "train", {"benign": 1, "malignant": 1})
    spec = data()
    found = discover_labelled(root, spec)
    dataset = ClassificationDataset(found.samples, model(), spec, only_images=True)
    assert dataset[0][1] is None


def test_augmentation_moves_the_image_and_not_the_label(tmp_path):
    """A flip does not move an image's label, which is why the augmenter is called with no
    mask at all - and why this task needs no vector-safe transform list."""
    from pyplatypus.data.augmentation import build_augmenter
    from pyplatypus.spec.components import AugmentationStep

    root = tree(tmp_path / "train", {"benign": 1, "malignant": 1}, size=32)
    spec = data()
    found = discover_labelled(root, spec)
    plain = ClassificationDataset(found.samples, model(), spec)
    flipped = ClassificationDataset(
        found.samples,
        model(),
        spec,
        augmenter=build_augmenter([AugmentationStep(name="HorizontalFlip", params={"p": 1.0})]),
    )
    assert plain[0][1] == flipped[0][1]
    assert not np.allclose(plain[0][0], flipped[0][0])
    assert np.allclose(plain[0][0], flipped[0][0][:, ::-1])


# --------------------------------------------------------------------------- the loader


def test_the_loader_hands_torch_a_channels_first_image_and_a_long_label(tmp_path):
    root = tree(tmp_path / "train", {"benign": 4, "malignant": 4})
    spec = data()
    dataset = ClassificationDataset(discover_labelled(root, spec).samples, model(), spec)
    images, labels = next(iter(make_classification_loader(dataset, batch_size=4)))

    assert images.shape == (4, 3, 16, 16)
    assert labels.shape == (4,)
    assert labels.dtype == torch.long
    assert labels.tolist() == [0, 0, 0, 0]


def test_the_label_is_not_transposed_on_the_way_through(tmp_path):
    """The one target in this package that does not go through `to_channels_first`: it is a
    scalar, so there is no axis to move, and the one-hot shape somebody might reach for
    instead would transpose a batch silently."""
    root = tree(tmp_path / "train", {"benign": 2, "malignant": 2})
    spec = data()
    dataset = ClassificationDataset(discover_labelled(root, spec).samples, model(), spec)
    loader = make_classification_loader(dataset, batch_size=4)
    _, labels = next(iter(loader))
    assert labels.ndim == 1
    assert sorted(labels.tolist()) == [0, 0, 1, 1]


# ------------------------------------------------------------------------------- the join


def test_a_dataset_on_disk_reaches_a_head_and_a_metric(tmp_path):
    """Green on each half proves the halves. This is the way a user reaches it: pictures in
    directories, a loader, the head slice 3 built, and the metrics slice 4 built - with
    nothing between them that an engine has not been written for yet."""
    root = tree(tmp_path / "train", {"benign": 6, "malignant": 6}, size=28)
    spec = data()
    specification = model(blocks=2, filters=4)
    dataset = ClassificationDataset(discover_labelled(root, spec).samples, specification, spec)
    network = build_classifier(specification, n_class=spec.n_class)

    accuracy, auc = Accuracy(), Auc()
    seen = 0
    for images, labels in make_classification_loader(dataset, batch_size=4, shuffle=True):
        logits = network(images)
        assert logits.shape == (images.shape[0], spec.n_class)
        accuracy.update(logits, labels)
        auc.update(logits, labels)
        seen += images.shape[0]

    assert seen == 12
    assert 0.0 <= accuracy.compute() <= 1.0
    assert 0.0 <= auc.compute() <= 1.0
    assert dataset.balance().counts == {"benign": 6, "malignant": 6}
