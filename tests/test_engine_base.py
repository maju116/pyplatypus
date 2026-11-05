"""What the engines share, tested as a set rather than once per engine.

Two of them diverged twice. Detection recorded which splits carry annotations while
segmentation threw the same fact away, and the guard order that let "no masks" stand in for
"no split" was wrong in both and had to be fixed in both - the second time on the same night
the conditional was written twice. A test that asks the same question of each and compares
the answers is what was missing; `EngineBase` is what makes the answers come from one place.

**Three columns since `instance_segmentation` arrived**, which is the shape this file is
for: a third engine is a third column here rather than a second test file, because the
thing worth asserting is that the three answers are *the same answer* and a file per engine
cannot say that.

**The structural tests have four columns since `pyplatypus#320`, and since
`pyplatypus#388` so do the behavioural ones** - which is the paragraph that used to stand
here being paid rather than deleted. It said the behavioural columns *"are a piece of work
rather than a line, and they are owed"*, and the one that was missing was the instance
engine's; the tenth review pass found exactly that, by writing the name into a comment and
returning `True` with this file still green.

**So what the grep below can and cannot do is worth stating.** It asserts that each engine
names the shared function, which is what catches a copy of the body being reintroduced -
the divergence `#320` found twice. It cannot see a body at all, so it is satisfied by the
name in a comment. The question *does this engine answer from the layout* is asked on a
fixture, once per task, and those four tests are:

    Engine                test_a_test_split_without_masks_is_not
    DetectionEngine       test_a_test_split_without_annotations_can_be_predicted_but_not_scored
    ClassificationEngine  test_a_test_set_of_images_alone_is_read_and_predicted
    InstanceEngine        test_a_test_split_of_images_alone_reads_as_UNLABELLED_and_a_labelled_one_does_not

By name rather than by file and line, because a line number names the place somebody went
looking rather than the place the test is - and the four names are asserted to exist by
`test_each_engine_s_behavioural_column_exists` below, so this list cannot quietly become
prose about tests that were renamed.
"""

import inspect
from pathlib import Path

import pytest

from pyplatypus import build_engine
from pyplatypus.classification_engine import ClassificationEngine
from pyplatypus.detection_engine import DetectionEngine
from pyplatypus.engine import Engine, EngineBase, EngineError
from pyplatypus.instance_engine import InstanceEngine
from pyplatypus.spec.loader import from_dict


def test_every_engine_is_built_on_the_same_base():
    assert issubclass(Engine, EngineBase)
    assert issubclass(DetectionEngine, EngineBase)
    assert issubclass(InstanceEngine, EngineBase)


def test_each_says_what_a_labelled_split_carries_and_the_base_does_not_guess():
    """The one word the shared message needs, provided by the subclass. The base's own
    value is never used, which is what makes forgetting to set it visible."""
    assert Engine.labels_are == "masks"
    assert DetectionEngine.labels_are == "annotations"
    assert InstanceEngine.labels_are == "per-object masks"
    assert EngineBase.labels_are not in {"masks", "annotations", "per-object masks"}
    # Three distinct words, because the message they go into is the one that tells somebody
    # which files are missing - and "masks" for the instance engine would send them looking
    # for the wrong layout.
    assert len({Engine.labels_are, DetectionEngine.labels_are, InstanceEngine.labels_are}) == 3


def test_discovery_is_the_base_s_and_neither_engine_has_its_own():
    """`_discover_test` and `_require_split` were the same work written twice. Asserted on
    `__dict__` rather than `hasattr`, which would be satisfied by inheritance."""
    for engine in (Engine, DetectionEngine, InstanceEngine, ClassificationEngine):
        assert "_discover_test" not in engine.__dict__
        assert "_require_split" not in engine.__dict__
        # The hook is the exception: masks, annotations and a class name are found in
        # different ways, and one of the four answers before looking at the layout at all.
        assert "_has_any_labels" in engine.__dict__
    for engine in (Engine, DetectionEngine, InstanceEngine):
        assert "_discover_splits" not in engine.__dict__
    # Classification overrides that one deliberately rather than by drift: its split is
    # stratified, because strata are the decision that task turns on, and the three-paths
    # case goes to `super()`. Asserted rather than left out, so that the exception is a
    # statement and not a gap in the loop above.
    assert "_discover_splits" in ClassificationEngine.__dict__

    # And all four call one function rather than carrying a copy of it - which is the
    # divergence this file was written after, and which `pyplatypus#320` found twice more:
    # detection carried that body verbatim while the function's own docstring said
    # detection "answers this differently", and classification carried `return True`.
    from pyplatypus.engine import labels_in_test_split

    # A grep, and knowing that is the point: it catches a copy of the body coming back and
    # is satisfied by the name in a comment. The four behavioural columns named in this
    # file's docstring are what hold the answers. `pyplatypus#388`.
    for engine in (Engine, DetectionEngine, InstanceEngine, ClassificationEngine):
        source = inspect.getsource(engine._has_any_labels)
        assert labels_in_test_split.__name__ in source, engine.__name__
    # The control, because a grep over source that cannot fail proves nothing: the base's
    # own hook raises and names no such thing.
    assert labels_in_test_split.__name__ not in inspect.getsource(EngineBase._has_any_labels)


# --- the same question of both -------------------------------------------------------------


def _segmentation(config, nested_root, **data):
    config["data"]["train_path"] = str(nested_root)
    config["data"]["validation_path"] = str(nested_root)
    config["data"].update(data)
    if data.get("validation") is False:
        config["data"].pop("validation_path", None)
    return build_engine(from_dict(config), device="cpu")


def _instance(nested_root, **data):
    """The same `nested_dirs` layout the segmentation column uses.

    Which is the point rather than a convenience: the Data Science Bowl's layout is one
    image and several binary mask files per case, and that is `nested_root`. The two tasks
    *find* those files identically and differ only in what happens after they are read, so
    a fixture of its own here would be asserting that two layouts behave alike instead of
    that one layout is read by one discovery.
    """
    config = {
        "task": "instance_segmentation",
        "data": {
            "train_path": str(nested_root),
            "validation_path": str(nested_root),
            **data,
        },
        "models": [
            {"name": "nuclei", "input_shape": [64, 64], "separation": {"name": "rim"}, "blocks": 2}
        ],
    }
    if data.get("validation") is False:
        config["data"].pop("validation_path", None)
    return build_engine(from_dict(config), device="cpu")


def _detection(detection_config, detection_root, **data):
    detection_config["data"]["train_path"] = str(detection_root / "train")
    detection_config["data"]["validation_path"] = str(detection_root / "valid")
    detection_config["data"].update(data)
    if data.get("validation") is False:
        detection_config["data"].pop("validation_path", None)
    return build_engine(from_dict(detection_config), device="cpu")


def test_every_engine_records_the_same_splits_as_labelled(
    config, nested_root, detection_config, detection_root
):
    engines = [
        _segmentation(config, nested_root),
        _instance(nested_root),
        _detection(detection_config, detection_root),
    ]
    for engine in engines:
        assert engine.labelled == {"train", "validation"}
        assert set(engine._samples) == {"train", "validation"}


def test_every_engine_drops_validation_when_the_run_says_it_has_none(
    config, nested_root, detection_config, detection_root
):
    for engine in (
        _segmentation(config, nested_root, validation=False),
        _instance(nested_root, validation=False),
        _detection(detection_config, detection_root, validation=False),
    ):
        assert "validation" not in engine.labelled
        assert "validation" not in engine._samples
        assert engine.labelled == {"train"}


def test_every_engine_refuses_a_missing_split_with_the_same_sentence(
    config, nested_root, detection_config, detection_root
):
    """The divergence that mattered: one said "no masks" where the truth was "no split",
    and the other said "no annotations" for the same reason. One message now."""
    for engine in (
        _segmentation(config, nested_root, validation=False),
        _instance(nested_root, validation=False),
        _detection(detection_config, detection_root, validation=False),
    ):
        with pytest.raises(EngineError, match="`validation: false`"):
            engine._require_split("validation")
        with pytest.raises(EngineError, match="no 'nonsense' data in this spec"):
            engine._require_split("nonsense")


def test_every_engine_names_what_is_missing_before_naming_what_is_unlabelled(
    config, nested_root, detection_config, detection_root
):
    """Order, which was wrong in two of them. A split that was never created is a different
    thing from one that exists without labels, and the second message sends somebody
    looking for files they never wrote."""
    one = _segmentation(config, nested_root, validation=False)
    three = _instance(nested_root, validation=False)
    two = _detection(detection_config, detection_root, validation=False)

    for engine, ask in (
        (one, "_needs_masks"),
        (three, "_needs_masks"),
        (two, "_needs_annotations"),
    ):
        with pytest.raises(EngineError, match="`validation: false`"):
            getattr(engine, ask)("validation")

    # And when the split does exist without labels, each names its own kind.
    for engine, ask, said in (
        (one, "_needs_masks", "has no masks"),
        (three, "_needs_masks", "has no per-object masks"),
        (two, "_needs_annotations", "has no annotations"),
    ):
        engine.labelled.discard("train")
        with pytest.raises(EngineError, match=said):
            getattr(engine, ask)("train")


def test_no_name_means_the_first_model_in_every_engine(
    config, nested_root, detection_config, detection_root
):
    """Nine methods documented `model_name` as defaulting to the first and none did.

    Asked of the resolution rather than of nine methods, because that is where it lives and
    because the alternative is a second detection fit - 61.5 million parameters on the
    processor - to prove something that is one lookup. One method is exercised end to end
    below.
    """
    for engine in (
        _segmentation(config, nested_root),
        _instance(nested_root),
        _detection(detection_config, detection_root),
    ):
        assert engine.runs == {}, "nothing is trained yet, so there is no first model"
        with pytest.raises(EngineError, match="no model has been trained or loaded yet"):
            engine._resolve_model(None)

        engine.runs = {"first": object(), "second": object()}
        assert engine._resolve_model(None) == "first"
        assert engine._resolve_model("second") == "second"


def test_a_method_called_with_no_model_name_answers_about_the_first(config, nested_root):
    """End to end, once: the signature's default and the resolution meeting in a real call.

    `evaluate_cases()` with no name has to give what naming the model gives. A default that
    resolved to something else - or a method that took the default and then looked up `None`
    anyway - would pass the structural test in `test_documented_arguments_exist.py` and fail
    here.
    """
    engine = _segmentation(config, nested_root)
    engine.fit()
    name = next(iter(engine.runs))
    assert [row["case"] for row in engine.evaluate_cases()] == [
        row["case"] for row in engine.evaluate_cases(name)
    ]


def test_each_engine_s_behavioural_column_exists():
    """The four tests this file's docstring names, asserted to be there.

    The docstring is the only place that says where each engine's answer is held, and
    `pyplatypus#388` is what happens when such a list is believed: the grep above read as
    the guard for four engines while one of them had no behavioural test at all. A name in
    a docstring is prose; this is the line that makes it a claim.
    """
    columns = {
        "tests/test_labelled_splits.py": "test_a_test_split_without_masks_is_not",
        "tests/test_detection_engine.py": (
            "test_a_test_split_without_annotations_can_be_predicted_but_not_scored"
        ),
        "tests/test_classification_engine.py": (
            "test_a_test_set_of_images_alone_is_read_and_predicted"
        ),
        "tests/test_instance_engine.py": (
            "test_a_test_split_of_images_alone_reads_as_UNLABELLED_and_a_labelled_one_does_not"
        ),
    }
    here = Path(__file__).resolve().parent.parent
    for file, name in columns.items():
        source = (here / file).read_text()
        assert f"def {name}(" in source, f"{file} has no {name}"
        # And the docstring above has to carry it, which is the half that rots.
        assert name in __doc__, f"this file's docstring does not name {name}"
