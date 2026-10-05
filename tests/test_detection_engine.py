"""A detection specification, trained.

These tests train a real YOLOv3 for two epochs on twelve synthetic images, which is
useless as a model and exactly right as a test: it exercises the whole path - fit anchors,
survey the targets, build the network, three target tensors through the loss, decode,
suppress, invert the letterbox, score - and that path is where a wiring mistake lives.
Nothing here asserts that the model learned anything, because it cannot have.

The one assertion that is about *correctness* rather than plumbing is
`test_predictions_come_back_in_the_source_images_own_pixels`, which hands the engine
constructed logits and checks the box comes back where it was put in. Everything else in
this file would pass with the letterbox inverted the wrong way round.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyplatypus import DetectionEngine, Engine, build_engine, from_dict
from pyplatypus.engine import EngineError


@pytest.fixture
def engine(detection_config):
    spec = from_dict(detection_config)
    return build_engine(spec, device="cpu")


@pytest.fixture(scope="module")
def trained(tmp_path_factory, make_detection_split):
    """One trained engine for every test that only reads from it.

    Module-scoped because training a 61.5-million-parameter detector on the processor
    costs about six seconds whatever the input size - the network's depth dominates, not
    the image, measured at 5.9s for 64x64 and 6.9s for 128x128 - so thirteen tests asking
    the same questions of thirteen identical models made this file take 130 seconds. The
    tests that are *about* fitting still fit for themselves, below.

    The fit runs under `simplefilter("error")`, so an ordinary run that starts warning
    about anything fails here and takes the file with it. The survey's warning is supposed
    to mean something, and a warning nobody checks for stops meaning anything.
    """
    root = tmp_path_factory.mktemp("detection_shared")
    make_detection_split(root / "train", "train", 8, seed=0)
    make_detection_split(root / "valid", "valid", 4, seed=1)

    spec = from_dict({
        "task": "detection", "seed": 1,
        "data": {"train_path": str(root / "train"),
                 "validation_path": str(root / "valid"),
                 "classes": ["square", "bar"]},
        "models": [{"name": "d", "input_shape": [128, 128], "epochs": 2,
                    "batch_size": 2, "anchors_per_grid": 2,
                    "optimizer": {"name": "adam", "learning_rate": 1e-3}}],
    })
    engine = build_engine(spec, device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        engine.fit()
    return engine


# --- which engine ------------------------------------------------------------------------

def test_build_engine_follows_the_task(detection_config, config, nested_root):
    """Holding a spec is enough; the caller never has to ask which task it is."""
    assert isinstance(build_engine(from_dict(detection_config), device="cpu"),
                      DetectionEngine)

    config["data"]["train_path"] = str(nested_root)
    config["data"]["validation_path"] = str(nested_root)
    assert isinstance(build_engine(from_dict(config)), Engine)


def test_each_engine_refuses_the_other_task(detection_config, config):
    with pytest.raises(EngineError, match="trains segmentation"):
        Engine(from_dict(detection_config))
    with pytest.raises(EngineError, match="trains detectors"):
        DetectionEngine(from_dict(config))


def test_the_splits_are_found(engine):
    assert engine.split_sizes() == {"train": 8, "validation": 4}
    assert engine.labelled == {"train", "validation"}


# --- anchors -----------------------------------------------------------------------------

def test_anchors_are_fitted_to_the_training_boxes(trained):
    """Unset in the spec means fitted, and the fit is recorded on the run - which is the
    only place they exist afterwards, and the thing a detector cannot be reloaded without.
    """
    run = trained.runs["d"]
    assert run.anchor_fit is not None
    assert run.anchor_fit.boxes_used == 16
    assert 0.5 < run.anchor_fit.mean_iou <= 1.0
    assert len(run.anchors) == 3
    assert all(len(group) == 2 for group in run.anchors)
    assert all(0 < w <= 1 and 0 < h <= 1 for group in run.anchors for w, h in group)


def test_anchors_given_in_the_spec_are_not_refitted(detection_config):
    """A spec that names anchors is reproducible from the spec alone, so nothing should
    quietly replace them - and `anchor_fit` staying None is how a reader can tell which
    of the two happened."""
    given = [[[0.30, 0.30], [0.20, 0.20]],
             [[0.16, 0.16], [0.12, 0.12]],
             [[0.08, 0.08], [0.05, 0.05]]]
    detection_config["models"][0]["anchors"] = given
    engine = build_engine(from_dict(detection_config), device="cpu")
    engine.fit()

    run = engine.runs["d"]
    assert run.anchor_fit is None
    assert run.anchors == tuple(tuple(tuple(p) for p in g) for g in given)


def test_anchor_coverage_can_be_asked_of_a_split_they_were_not_fitted_on(trained):
    """The question worth asking: anchors covering the training boxes at 0.92 and the
    validation boxes at 0.70 say the two splits hold different objects, and no training
    curve shows that."""
    coverage = trained.anchor_coverage("d", "validation")
    assert coverage["boxes"] == 8
    assert coverage["anchors"] == 6
    assert 0 < coverage["mean_iou"] <= 1


# --- the survey --------------------------------------------------------------------------

def test_the_targets_are_surveyed_before_training(trained):
    survey = trained.runs["d"].survey
    assert survey is not None
    assert survey.images == 8
    assert survey.total == 16
    assert survey.placed + survey.unplaced == survey.total


def test_a_dataset_the_encoder_cannot_represent_warns(detection_config, detection_root,
                                                      voc_sample):
    """Two boxes of one shape in one cell share a slot, so the second is never shown to
    the model and never counted as missed. Warned rather than refused: it is a property of
    the data meeting the input size, and the fix is the user's choice of either."""
    # Enough distinct shapes that fitting six anchors is possible - k-means refuses
    # otherwise, correctly - with one colliding pair in every image.
    crowded = detection_root / "crowded"
    for n in range(4):
        voc_sample(crowded, f"c_{n}",
                   [(40, 40, 64 + n, 64 + n), (42, 42, 66 + n, 66 + n),
                    (10, 90, 30 + n, 110), (70, 10, 100, 24 + n)],
                   [0, 0, 0, 1])
    detection_config["data"]["train_path"] = str(crowded)
    engine = build_engine(from_dict(detection_config), device="cpu")

    with pytest.warns(UserWarning, match="cannot hold"):
        engine.fit()


# --- training ----------------------------------------------------------------------------

def test_fit_reports_the_four_parts_for_train_and_validation(trained):
    """A YOLOv3 total means nothing alone: it has a floor above zero that depends on the
    data, so `coordinates 5.93 / objectness 0.001` is the difference between a model that
    has learned what is where and one that has stalled."""
    history = trained.runs["d"].history
    assert len(history) == 2
    columns = set(history.columns)
    for prefix in ("train", "val"):
        assert {f"{prefix}_{part}" for part in
                ("loss", "coordinates", "objectness", "no_object", "classes")} <= columns


def test_a_callback_watching_the_validation_loss_works(detection_config):
    """Detection reuses the segmentation callbacks unchanged, which is only true if the
    quantity they watch is actually produced."""
    detection_config["models"][0]["epochs"] = 3
    detection_config["models"][0]["callbacks"] = [
        {"name": "early_stopping", "monitor": "val_loss", "patience": 1}
    ]
    engine = build_engine(from_dict(detection_config), device="cpu")
    history = engine.fit()["d"]
    assert len(history) <= 3


def test_accumulation_does_not_change_the_shape_of_a_run(detection_config):
    """Gradient accumulation is a memory trade, not a different experiment. The run has
    to look the same from outside; only batch norm sees the difference."""
    engine = build_engine(from_dict(detection_config), device="cpu", accumulate=2)
    history = engine.fit()["d"]
    assert len(history) == 2
    assert history.records[0]["train_loss"] > 0


# --- scoring -----------------------------------------------------------------------------

def test_the_comparison_table_has_detections_columns(trained):
    row = trained.evaluate("validation")[0]
    assert row["model"] == "d"
    assert row["architecture"] == "yolo3"
    assert row["n_truth"] == 8
    for key in ("map_50", "map_50_95", "mean_matched_iou"):
        assert key in row


def test_the_table_carries_no_overall_precision_or_recall(trained):
    """Averaging them over classes needs a weighting, and every weighting is a different
    claim. On BCCD - 4155 red cells to 372 white - a single precision is a statement about
    red cells wearing the costume of a statement about the model."""
    row = trained.evaluate("validation")[0]
    assert "precision" not in row
    assert "recall" not in row
    per_class = trained.evaluate_classes("d", "validation")
    assert all("precision" in entry and "recall" in entry for entry in per_class)


def test_one_row_per_class(trained, detection_classes):
    rows = trained.evaluate_classes("d", "validation")
    assert [row["class"] for row in rows] == detection_classes
    assert sum(row["n_truth"] for row in rows) == 8


def test_one_row_per_image_keyed_by_the_sample(trained):
    """The question a table of averages cannot answer: *which* images it fails on."""
    rows = trained.evaluate_images("d", "validation")
    assert len(rows) == trained.split_sizes()["validation"]

    keys = [row["key"] for row in rows]
    assert len(set(keys)) == len(keys)          # a row you can find again
    assert sum(row["n_truth"] for row in rows) == 8   # the split's own count

    for row in rows:
        assert row["missed"] == row["n_truth"] - row["matched"]
        assert row["spurious"] == row["n_predicted"] - row["matched"]


def test_no_average_precision_per_image(trained):
    """AP is the area under a precision-recall curve, so it is a property of a ranking
    over a dataset. On one image with three boxes it moves on a single box's rank and
    means nothing - the same trap as a per-case Dice on an empty mask."""
    row = trained.evaluate_images("d", "validation")[0]
    assert "average_precision" not in row and "map_50" not in row
    assert "mean_matched_iou" in row


def test_the_per_image_counts_decompose_the_table(trained):
    """Read at the same threshold, the rows and the per-class table must agree about how
    many boxes there were. If they ever disagree, someone chasing a dataset-wide number
    through the per-image rows is chasing a difference in the code."""
    rows = trained.evaluate_images("d", "validation")
    per_class = trained.evaluate_classes("d", "validation")
    assert sum(r["n_truth"] for r in rows) == sum(c["n_truth"] for c in per_class)
    assert sum(r["n_predicted"] for r in rows) == sum(c["n_predicted"] for c in per_class)


def test_the_default_threshold_is_the_operating_point(trained):
    """Counts need a confidence. The default is the one the specification names, and a
    looser threshold can only add predictions."""
    default = trained.evaluate_images("d", "validation")
    loose = trained.evaluate_images("d", "validation", score_threshold=0.0)
    assert sum(r["n_predicted"] for r in loose) >= sum(r["n_predicted"] for r in default)


def test_per_image_scoring_refuses_a_split_without_annotations(trained, monkeypatch):
    """A zero in a table reads as a result; "there is nothing to compare against" is a
    different statement and has to be said out loud."""
    monkeypatch.setattr(trained, "labelled", {"train"})
    with pytest.raises(EngineError, match="no annotations"):
        trained.evaluate_images("d", "validation")


def test_one_report_serves_both_tables(trained, monkeypatch):
    """`evaluate` and `evaluate_classes` are views over `report`, so a caller that wants
    both runs the model over the split once. Counted rather than assumed, because the
    saving is the only reason `report` is public."""
    passes = []
    original = trained.predict
    monkeypatch.setattr(trained, "predict",
                        lambda *a, **k: (passes.append(1), original(*a, **k))[1])

    measured = trained.report("d", "validation")
    row = measured.as_row(trained.runs["d"])
    per_class = measured.per_class()

    assert len(passes) == 1
    assert row["map_50"] == trained.evaluate("validation")[0]["map_50"]
    assert per_class == trained.evaluate_classes("d", "validation")


def test_evaluating_before_training_says_so(engine):
    with pytest.raises(EngineError, match="call fit"):
        engine.evaluate("validation")


def test_an_unknown_model_lists_the_ones_there_are(trained):
    with pytest.raises(EngineError, match="trained so far: d"):
        trained.predict("nope")


def test_best_model_refuses_a_column_that_does_not_exist(trained):
    with pytest.raises(EngineError, match="no column 'dice'"):
        trained.best_model("dice")


def test_best_model_ranks_on_average_precision(trained):
    assert trained.best_model("map_50", "validation") == "d"


# --- predictions -------------------------------------------------------------------------

def test_predict_returns_one_entry_per_image_with_names(trained):
    predictions = trained.predict("d", "validation")
    assert isinstance(predictions, list)
    assert len(predictions) == 4
    first = predictions[0]
    assert first["key"].startswith("valid_")
    assert first["boxes"].shape[1] == 4
    assert len(first["scores"]) == len(first["boxes"]) == len(first["labels"])
    assert all(name in ("square", "bar") for name in first["names"])


def test_predictions_come_back_in_the_source_images_own_pixels(trained, monkeypatch):
    """The correspondence test, and the only one here that could fail while every shape
    stays right.

    The model is handed the logits that encode one known box, so what `decode` produces is
    determined rather than learned. If the inverse letterbox is missing, inverted, or
    applied to the wrong axis, the box comes back in the network's 128x128 frame instead of
    the photograph's 160x128 - still four plausible numbers, in the wrong place.
    """
    from pyplatypus.detection.encode import encode

    run = trained.runs["d"]
    dataset = trained.dataset(run.spec, "validation", anchors=run.anchors)
    example = dataset.read(0)
    source_box = example.annotation.boxes[0]

    encoded = encode(example.boxes[:1], example.labels[:1], anchors=run.anchors,
                     input_shape=(128, 128), n_class=2)

    def logits_for_that_box(_image):
        """The raw outputs a model would have to emit to reproduce this target exactly.

        12.0 rather than infinity for the certainties, which is why the encoder stores the
        within-cell offset and not its logit.
        """
        out = []
        for target in encoded.targets:
            grid = np.zeros_like(target)
            has_object = target[..., 4] > 0.5
            offsets = np.clip(target[..., 0:2], 1e-6, 1 - 1e-6)
            grid[..., 0:2] = np.log(offsets / (1 - offsets))
            grid[..., 2:4] = target[..., 2:4]
            grid[..., 4] = np.where(has_object, 12.0, -12.0)
            grid[..., 5:] = np.where(target[..., 5:] > 0.5, 12.0, -12.0)
            out.append(grid)
        return out

    monkeypatch.setattr(run.trainer, "raw_outputs", logits_for_that_box)
    predictions = trained.predict("d", "validation")

    found = predictions[0]["boxes"]
    assert len(found) == 1, "one box went in, so one box should come out"
    assert found[0] == pytest.approx(source_box, abs=1.0)
    # And the proof it is not simply the network's frame: a 160-wide source letterboxed
    # into 128 means source coordinates run past 128, which model-space ones cannot.
    assert example.annotation.width == 160
    assert example.boxes[0][2] < 128


def test_a_test_split_without_annotations_can_be_predicted_but_not_scored(
        detection_config, detection_root, voc_sample):
    """Both halves matter. Scoring an unlabelled split against empty truth would report
    zero, which is a different statement from "there is nothing to score against"."""
    held_out = detection_root / "test"
    for n in range(3):
        sample = voc_sample(held_out, f"t_{n}", [(20, 24, 56, 60)], [0])
        (sample / "annotations" / f"t_{n}.xml").unlink()
    detection_config["data"]["test_path"] = str(held_out)

    engine = build_engine(from_dict(detection_config), device="cpu")
    assert engine.labelled == {"train", "validation"}
    engine.fit()

    assert len(engine.predict("d", "test")) == 3
    with pytest.raises(EngineError, match="no annotations"):
        engine.evaluate("test")


def test_a_test_split_with_some_annotations_missing_is_an_error(
        detection_config, detection_root, voc_sample):
    """Not the same thing as an unlabelled split, and the difference matters.

    Falling back to images-only whenever labelled discovery failed would turn a split with
    one missing file into an unlabelled one, throwing away the annotations that are there
    and reporting nothing. The incomplete sample is named instead.
    """
    held_out = detection_root / "test_partial"
    for n in range(3):
        sample = voc_sample(held_out, f"t_{n}", [(20, 24, 56, 60)], [0])
        if n == 1:
            (sample / "annotations" / f"t_{n}.xml").unlink()
    detection_config["data"]["test_path"] = str(held_out)

    from pyplatypus import ConfigError

    with pytest.raises(ConfigError, match="t_1"):
        build_engine(from_dict(detection_config), device="cpu")


# --- augmentation -------------------------------------------------------------------------

def test_training_is_augmented_and_validation_is_not(detection_config):
    """Measuring a model on distorted data measures the distortion, so validation is never
    augmented. Asserted on the loaders the trainer is actually handed, because a spec that
    asks for augmentation and a pipeline that applies it are two different things."""
    detection_config["models"][0]["augmentation"] = [
        {"name": "HorizontalFlip", "params": {"p": 1.0}}
    ]
    engine = build_engine(from_dict(detection_config), device="cpu")
    model = engine.spec.models[0]

    train = engine.loader(model, "train", augmented=True)
    validation = engine.loader(model, "validation")
    assert train.dataset.base.augmenter is not None
    assert validation.dataset.base.augmenter is None


def test_augmentation_moves_the_boxes_with_the_image(detection_config):
    """The correspondence assertion again, this time through the engine: the fixture paints
    each box bright, so after a flip the pixels inside the transformed box must still be
    the bright ones. A flip applied to the image and not to the boxes passes every shape
    check and trains a model to find blood cells in empty background."""
    detection_config["models"][0]["augmentation"] = [
        {"name": "HorizontalFlip", "params": {"p": 1.0}}
    ]
    engine = build_engine(from_dict(detection_config), device="cpu")
    model = engine.spec.models[0]

    plain = engine.dataset(model, "train")
    flipped = engine.dataset(model, "train", augmented=True)

    image, _ = flipped[0]
    reference = plain.read(0)
    assert not np.allclose(image, reference.image), "nothing was augmented"

    # Where the flipped boxes must be, and what the pixels there must look like.
    width = image.shape[1]
    for box in reference.boxes:
        x0, y0, x1, y1 = box
        moved = (width - x1, y0, width - x0, y1)
        patch = image[int(moved[1]) + 2:int(moved[3]) - 2,
                      int(moved[0]) + 2:int(moved[2]) - 2, 0]
        assert patch.size > 0
        assert patch.mean() > 0.4, "the box no longer covers its object"


def test_a_transform_that_cannot_handle_boxes_is_named(detection_config):
    """albumentations raises from inside itself, so the refusal has to come from here. The
    probe runs at the model's input size - a fixed-size probe refused a legitimate crop,
    which is the check failing for its own reasons."""
    detection_config["models"][0]["augmentation"] = [
        {"name": "RandomCrop", "params": {"height": 512, "width": 512, "p": 1.0}}
    ]
    engine = build_engine(from_dict(detection_config), device="cpu")

    from pyplatypus.data.augmentation import AugmentationError

    with pytest.raises(AugmentationError, match="RandomCrop"):
        engine.dataset(engine.spec.models[0], "train", augmented=True)


def test_augmentation_survives_the_worker_processes(detection_config):
    """The combination the real BCCD run uses, and the one nothing else here covers.

    Every other test runs with `num_workers=0`, so the augmenter lives in the main
    process. With workers it is inherited on Linux and pickled on Windows, and an
    albumentations pipeline that could not cross would fail only there - which is how the
    `unplaced` counters were lost in §4p: they were incremented in workers the main
    process never heard from.
    """
    detection_config["models"][0]["augmentation"] = [
        {"name": "HorizontalFlip", "params": {"p": 0.5}}
    ]
    engine = build_engine(from_dict(detection_config), device="cpu", num_workers=2)
    history = engine.fit()["d"]
    assert len(history) == 2
    assert len(engine.predict("d", "validation")) == 4


def test_a_detector_trains_with_augmentation(detection_config):
    """End to end, because the augmenter is built inside the loader and every earlier
    assertion here is about the pieces."""
    detection_config["models"][0]["augmentation"] = [
        {"name": "HorizontalFlip", "params": {"p": 0.5}},
        {"name": "RandomBrightnessContrast", "params": {"p": 0.5}},
    ]
    engine = build_engine(from_dict(detection_config), device="cpu")
    history = engine.fit()["d"]
    assert len(history) == 2
    assert history.records[-1]["train_loss"] > 0


# --- the picture of the anchor fit --------------------------------------------------------

def test_box_shapes_returns_the_cloud_and_the_anchors_in_one_frame(trained,
                                                                   detection_classes):
    """Everything a plot of the anchor fit needs, in one call and one set of coordinates.
    Widths and heights from one place and anchors from another is how a picture comes to
    show boxes in different places from where the anchors were fitted to them."""
    shapes = trained.box_shapes("d", "train")

    assert shapes["classes"] == detection_classes
    assert shapes["input_shape"] == [128, 128]
    assert shapes["anchors_were_fitted"] is True
    assert np.asarray(shapes["anchors"]).shape == (3, 2, 2)

    boxes = shapes["boxes"]
    assert len(boxes["width"]) == 16        # eight images, two boxes each
    assert set(boxes["name"]) == set(detection_classes)
    assert all(0 < w <= 1 for w in boxes["width"])
    assert all(0 < h <= 1 for h in boxes["height"])


def test_the_cloud_and_the_anchors_are_in_the_same_coordinates(trained):
    """The property the whole thing rests on. Both are fractions of the model's input,
    computed by the function that fitted the anchors - so an anchor sitting among its
    boxes on the picture really is sitting among them."""
    shapes = trained.box_shapes("d", "train")
    cloud = np.stack([shapes["boxes"]["width"], shapes["boxes"]["height"]], axis=1)
    anchors = np.asarray(shapes["anchors"]).reshape(-1, 2)

    # Every anchor is within the cloud's range, because k-means puts centres among their
    # points. If the two were in different coordinates this would fail by a wide margin.
    assert anchors[:, 0].min() >= cloud[:, 0].min() * 0.5
    assert anchors[:, 0].max() <= cloud[:, 0].max() * 2.0
    assert anchors[:, 1].min() >= cloud[:, 1].min() * 0.5
    assert anchors[:, 1].max() <= cloud[:, 1].max() * 2.0


def test_box_shapes_refuses_a_split_with_no_annotations(trained, detection_config,
                                                        detection_root, voc_sample):
    """A cloud of boxes needs boxes."""
    held_out = detection_root / "unlabelled"
    for n in range(2):
        sample = voc_sample(held_out, f"u_{n}", [(20, 24, 56, 60)], [0])
        (sample / "annotations" / f"u_{n}.xml").unlink()
    detection_config["data"]["test_path"] = str(held_out)

    engine = build_engine(from_dict(detection_config), device="cpu")
    engine.fit()
    with pytest.raises(EngineError, match="no annotations"):
        engine.box_shapes("d", "test")


# --- weights -----------------------------------------------------------------------------

def test_exported_weights_carry_the_anchors(trained, tmp_path):
    """Not optional metadata. The same weights read with other anchors decode every box
    scaled by a fixed factor - plausible boxes, plausible scores, wrong places."""
    import json

    written = trained.export_weights("d", tmp_path / "d.safetensors")
    sidecar = json.loads(written.with_suffix(".json").read_text())

    assert sidecar["architecture"] == "yolo3"
    assert sidecar["anchors_per_grid"] == 2
    assert sidecar["classes"] == ["square", "bar"]
    assert np.asarray(sidecar["anchors"]).shape == (3, 2, 2)
    assert sidecar["anchors"] == [[list(pair) for pair in group]
                                  for group in trained.runs["d"].anchors]
    # `blocks` and `filters` identify a U-shaped model and say nothing about a detector.
    assert "blocks" not in sidecar
    assert "n_class" not in sidecar


def test_a_reloaded_detector_predicts_the_same_boxes(trained, detection_config,
                                                     tmp_path):
    """The claim that matters, and the reason loading adopts the anchors from the file.

    Export, then load into an engine whose specification names **no** anchors at all, and
    the predictions must be the ones the trained model made. Read with any other anchors
    the same weights decode every box scaled by a fixed factor - plausible boxes, plausible
    scores, wrong places - so comparing the boxes is what distinguishes "loaded" from
    "loaded and silently wrong". Verified by scaling the adopted anchors by 1.2: the box
    comparison fails on its own, with the anchor assertion above removed.
    """
    written = trained.export_weights("d", tmp_path / "d.safetensors")
    before = trained.predict("d", "validation")

    detection_config["data"] = {
        "train_path": trained.spec.data.train_path,
        "validation_path": trained.spec.data.validation_path,
        "classes": list(trained.spec.data.classes),
    }
    detection_config["models"][0] = {
        **detection_config["models"][0], "weights": str(written), "fit": False,
    }
    reloaded = build_engine(from_dict(detection_config), device="cpu")
    reloaded.fit()

    assert reloaded.runs["d"].anchors == trained.runs["d"].anchors
    assert reloaded.runs["d"].anchor_fit is None, "nothing should have been refitted"
    assert reloaded.runs["d"].trained is False

    after = reloaded.predict("d", "validation")
    assert len(after) == len(before)
    for one, other in zip(before, after, strict=True):
        assert one["key"] == other["key"]
        assert other["boxes"] == pytest.approx(one["boxes"], abs=1e-4)
        assert other["scores"] == pytest.approx(one["scores"], abs=1e-5)
        assert list(other["labels"]) == list(one["labels"])


def test_weights_without_anchors_are_refused(trained, detection_config, tmp_path):
    """A detector without its anchors cannot be used at all, and there is no sensible
    default to fall back on - so a file that does not carry them is refused rather than
    guessed at. The case this covers is weights converted from somewhere else."""
    import json

    written = trained.export_weights("d", tmp_path / "bare.safetensors")
    sidecar = written.with_suffix(".json")
    payload = json.loads(sidecar.read_text())
    del payload["anchors"]
    sidecar.write_text(json.dumps(payload))

    detection_config["data"] = {
        "train_path": trained.spec.data.train_path,
        "validation_path": trained.spec.data.validation_path,
        "classes": list(trained.spec.data.classes),
    }
    detection_config["models"][0] = {
        **detection_config["models"][0], "weights": str(written), "fit": False,
    }
    engine = build_engine(from_dict(detection_config), device="cpu")
    with pytest.raises(EngineError, match="carries no anchors"):
        engine.fit()


def test_weights_trained_on_differently_named_classes_are_refused(trained,
                                                                  detection_config,
                                                                  tmp_path):
    """Same count, different meaning. This loads cleanly into torch and labels every box
    wrongly, which is the detection counterpart of weights trained on another colormap:
    the shapes agree and the answer is nonsense."""
    from pyplatypus.weights import WeightsError

    written = trained.export_weights("d", tmp_path / "d.safetensors")

    detection_config["data"] = {
        "train_path": trained.spec.data.train_path,
        "validation_path": trained.spec.data.validation_path,
        "classes": ["bar", "square"],          # the same two, swapped
    }
    detection_config["models"][0] = {
        **detection_config["models"][0], "weights": str(written), "fit": False,
    }
    engine = build_engine(from_dict(detection_config), device="cpu")
    with pytest.raises(WeightsError) as caught:
        engine.fit()
    # Named, with both orders shown: "a different model" alone would send someone looking
    # at the architecture.
    assert "classes: weights say ('square', 'bar')" in str(caught.value)
    assert "the model says ('bar', 'square')" in str(caught.value)


def test_naming_both_weights_and_anchors_is_refused(detection_config, tmp_path):
    """Two claims about one model with one of them untrue. Refused while the spec is read,
    so it costs nothing and cannot be reached by accident."""
    from pyplatypus import ConfigError

    detection_config["models"][0] = {
        **detection_config["models"][0],
        "weights": str(tmp_path / "x.safetensors"),
        "anchors": [[[0.3, 0.3], [0.2, 0.2]]] * 3,
    }
    with pytest.raises(ConfigError, match="both `weights` and `anchors`"):
        from_dict(detection_config)


