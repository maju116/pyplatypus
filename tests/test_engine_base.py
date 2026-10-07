"""What both engines share, tested as a pair rather than twice.

These two have diverged twice. Detection recorded which splits carry annotations while
segmentation threw the same fact away, and the guard order that let "no masks" stand in for
"no split" was wrong in both and had to be fixed in both - the second time on the same night
the conditional was written twice. A test that asks the same question of each and compares
the answers is what was missing; `EngineBase` is what makes the answers come from one place.
"""

import pytest

from pyplatypus import build_engine
from pyplatypus.detection_engine import DetectionEngine
from pyplatypus.engine import Engine, EngineBase, EngineError
from pyplatypus.spec.loader import from_dict


def test_both_engines_are_built_on_the_same_base():
    assert issubclass(Engine, EngineBase)
    assert issubclass(DetectionEngine, EngineBase)


def test_each_says_what_a_labelled_split_carries_and_the_base_does_not_guess():
    """The one word the shared message needs, provided by the subclass. The base's own
    value is never used, which is what makes forgetting to set it visible."""
    assert Engine.labels_are == "masks"
    assert DetectionEngine.labels_are == "annotations"
    assert EngineBase.labels_are not in {"masks", "annotations"}


def test_discovery_is_the_base_s_and_neither_engine_has_its_own():
    """`_discover_test` and `_require_split` were the same work written twice. Asserted on
    `__dict__` rather than `hasattr`, which would be satisfied by inheritance."""
    for engine in (Engine, DetectionEngine):
        assert "_discover_test" not in engine.__dict__
        assert "_require_split" not in engine.__dict__
        assert "_discover_splits" not in engine.__dict__
    # The hook is the exception: masks and annotations are found in different ways.
    assert "_has_any_labels" in Engine.__dict__
    assert "_has_any_labels" in DetectionEngine.__dict__


# --- the same question of both -------------------------------------------------------------


def _segmentation(config, nested_root, **data):
    config["data"]["train_path"] = str(nested_root)
    config["data"]["validation_path"] = str(nested_root)
    config["data"].update(data)
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


def test_both_record_the_same_splits_as_labelled(
    config, nested_root, detection_config, detection_root
):
    one = _segmentation(config, nested_root)
    two = _detection(detection_config, detection_root)
    assert one.labelled == two.labelled == {"train", "validation"}
    assert set(one._samples) == set(two._samples) == {"train", "validation"}


def test_both_drop_validation_when_the_run_says_it_has_none(
    config, nested_root, detection_config, detection_root
):
    one = _segmentation(config, nested_root, validation=False)
    two = _detection(detection_config, detection_root, validation=False)
    for engine in (one, two):
        assert "validation" not in engine.labelled
        assert "validation" not in engine._samples
        assert engine.labelled == {"train"}


def test_both_refuse_a_missing_split_with_the_same_sentence(
    config, nested_root, detection_config, detection_root
):
    """The divergence that mattered: one said "no masks" where the truth was "no split",
    and the other said "no annotations" for the same reason. One message now."""
    one = _segmentation(config, nested_root, validation=False)
    two = _detection(detection_config, detection_root, validation=False)
    for engine in (one, two):
        with pytest.raises(EngineError, match="`validation: false`"):
            engine._require_split("validation")
        with pytest.raises(EngineError, match="no 'nonsense' data in this spec"):
            engine._require_split("nonsense")


def test_both_name_what_is_missing_before_naming_what_is_unlabelled(
    config, nested_root, detection_config, detection_root
):
    """Order, which was wrong in both. A split that was never created is a different thing
    from one that exists without labels, and the second message sends somebody looking for
    files they never wrote."""
    one = _segmentation(config, nested_root, validation=False)
    two = _detection(detection_config, detection_root, validation=False)

    with pytest.raises(EngineError, match="`validation: false`"):
        one._needs_masks("validation")
    with pytest.raises(EngineError, match="`validation: false`"):
        two._needs_annotations("validation")

    # And when the split does exist without labels, each names its own kind.
    one.labelled.discard("train")
    two.labelled.discard("train")
    with pytest.raises(EngineError, match="has no masks"):
        one._needs_masks("train")
    with pytest.raises(EngineError, match="has no annotations"):
        two._needs_annotations("train")
