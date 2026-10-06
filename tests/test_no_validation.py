"""`validation: false`: a run that trains on everything and measures nothing.

A final fit once the hyperparameters are settled is a legitimate thing to want, and until
this existed it meant inventing a split and ignoring the number it produced. What is not
legitimate is arriving here by omission - "I have no validation set" and "I forgot" look
identical, and only one of them is a decision. So it is a field, and silence is still an
error.
"""

import warnings

import pytest

from pyplatypus import build_engine
from pyplatypus.engine import EngineError
from pyplatypus.spec.loader import ConfigError, from_dict


@pytest.fixture
def no_validation(config, nested_root):
    config["data"]["train_path"] = str(nested_root)
    config["data"].pop("validation_path", None)
    config["data"]["validation"] = False
    return config


# --- the specification ---------------------------------------------------------------------

def test_silence_is_still_an_error_and_the_message_names_the_third_option(config,
                                                                         nested_root):
    config["data"]["train_path"] = str(nested_root)
    config["data"].pop("validation_path", None)
    with pytest.raises(ConfigError, match="validation: false"):
        from_dict(config, check_paths=False)


def test_saying_both_where_it_comes_from_and_that_there_is_none_is_refused(config,
                                                                          nested_root):
    config["data"]["train_path"] = str(nested_root)
    config["data"]["validation_path"] = str(nested_root)
    config["data"]["validation"] = False
    with pytest.raises(ConfigError, match="cannot both be true"):
        from_dict(config, check_paths=False)

    config["data"].pop("validation_path")
    config["data"]["split"] = {"fractions": [0.8, 0.2], "group_by": None}
    with pytest.raises(ConfigError, match="cannot both be true"):
        from_dict(config, check_paths=False)


def test_a_callback_cannot_wait_for_a_number_that_will_never_arrive(no_validation):
    """Early stopping on `val_loss` with no validation trains to the last epoch while
    waiting, and checkpointing writes nothing. Both silently, which is why this is refused
    where both the model and the data are visible rather than in either alone."""
    no_validation["models"][0]["callbacks"] = [
        {"name": "early_stopping", "monitor": "val_loss", "patience": 2}
    ]
    with pytest.raises(ConfigError, match="would wait for one"):
        from_dict(no_validation, check_paths=False)


def test_watching_a_training_quantity_is_fine(no_validation):
    no_validation["models"][0]["callbacks"] = [
        {"name": "early_stopping", "monitor": "train_loss", "patience": 2}
    ]
    assert from_dict(no_validation, check_paths=False).data.validation is False


# --- the run -------------------------------------------------------------------------------

def test_it_trains_and_the_history_has_no_validation_columns(no_validation):
    engine = build_engine(from_dict(no_validation), device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        engine.fit()

    record = engine.runs[no_validation["models"][0]["name"]].history.records[-1]
    assert "train_loss" in record
    assert not [key for key in record if key.startswith("val_")], (
        "a run that measured nothing must not report a column for it"
    )


def test_the_validation_split_is_absent_rather_than_empty(no_validation):
    """An empty split makes every count read zero and every average read nan, which is a
    number for something that was never asked for."""
    engine = build_engine(from_dict(no_validation), device="cpu")
    assert "validation" not in engine._samples
    assert "validation" not in engine.labelled
    assert "train" in engine.labelled


def test_asking_for_the_validation_split_says_validation_false_and_not_no_masks(
        no_validation):
    """The two are different and the wrong one sends somebody looking for files. `evaluate`
    defaults to `"validation"`, so this is the first thing anyone meets after `fit`."""
    engine = build_engine(from_dict(no_validation), device="cpu")
    engine.fit()
    for call in (lambda: engine.evaluate(),
                 lambda: engine.evaluate_cases(no_validation["models"][0]["name"]),
                 lambda: engine.predict(no_validation["models"][0]["name"], "validation")):
        with pytest.raises(EngineError, match="`validation: false`"):
            call()


def test_everything_else_still_works_on_the_split_that_does_exist(no_validation):
    engine = build_engine(from_dict(no_validation), device="cpu")
    engine.fit()
    name = no_validation["models"][0]["name"]
    assert engine.evaluate("train")
    assert len(engine.predict(name, "train")) == len(engine._samples["train"])


# --- detection, which has the same three-way choice ------------------------------------------

def test_a_detector_can_also_be_fitted_without_validation(detection_config, detection_root):
    """`DetectionData` inherits the field, so the rule and the refusals come with it - but
    the engine is a separate class and had to be taught the same thing, which is the half a
    shared spec does not give for free."""
    detection_config["data"]["train_path"] = str(detection_root / "train")
    detection_config["data"].pop("validation_path", None)
    detection_config["data"]["validation"] = False
    detection_config["models"][0]["epochs"] = 1

    engine = build_engine(from_dict(detection_config), device="cpu")
    assert "validation" not in engine._samples
    assert "validation" not in engine.labelled

    engine.fit()
    record = engine.runs[detection_config["models"][0]["name"]].history.records[-1]
    assert not [key for key in record if key.startswith("val_")]

    with pytest.raises(EngineError, match="`validation: false`"):
        engine.evaluate()
