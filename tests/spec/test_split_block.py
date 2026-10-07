"""Dividing one folder instead of naming three, stated in the configuration.

`validation_path` was required and the refusal said only "Field required", which is true
and useless: the answer is a split, and nothing pointed at it. This is that answer written
where the configuration is.
"""

import pytest

from pyplatypus.errors import ConfigError
from pyplatypus.spec import from_dict

COLORMAP = [[0, 0, 0], [255, 255, 255]]
MODELS = [{"name": "m", "input_shape": [32, 32]}]


def build(**data):
    return from_dict(
        {
            "task": "semantic_segmentation",
            "data": {"train_path": "t", "colormap": COLORMAP, **data},
            "models": MODELS,
        },
        check_paths=False,
    )


def test_one_folder_and_a_split_is_enough():
    spec = build(split={"fractions": [0.8, 0.2], "group_by": None})
    assert spec.data.validation_path is None
    assert spec.data.split.fractions == (0.8, 0.2)


def test_neither_is_refused_and_the_refusal_names_the_way_out():
    """ "Field required" was true and useless. The answer is a split, so the refusal says so
    - and names `group_by` while it has the reader's attention, because that is the choice
    nobody knows they are making."""
    with pytest.raises(ConfigError) as caught:
        build()
    message = str(caught.value)
    assert "validation_path" in message and "split" in message
    assert "group_by" in message and "patient" in message


def test_both_is_refused():
    """Two sources for one thing are two places to change it, and one gets forgotten."""
    with pytest.raises(ConfigError, match="validate against"):
        build(validation_path="v", split={"fractions": [0.8, 0.2], "group_by": None})


def test_group_by_must_be_stated_even_to_say_no():
    """The field this whole block exists to put in front of someone.

    Slices of one patient in training and validation at once make validation measure memory
    rather than generalisation, and nothing in the output says so. An omitted key would be
    that choice made silently; `null` is the same choice made on purpose.
    """
    with pytest.raises(ConfigError, match="group_by"):
        build(split={"fractions": [0.8, 0.2]})


def test_two_or_three_fractions_and_the_rule_lives_in_one_place():
    assert build(split={"fractions": [0.8, 0.2], "group_by": None}) is not None
    assert build(split={"fractions": [0.6, 0.2, 0.2], "group_by": None}) is not None
    with pytest.raises(ConfigError, match="two or three fractions"):
        build(split={"fractions": [0.5, 0.2, 0.2, 0.1], "group_by": None})


def test_a_test_path_and_a_split_both_claim_the_test_set():
    with pytest.raises(ConfigError, match="test_path"):
        build(test_path="x", split={"fractions": [0.6, 0.2, 0.2], "group_by": None})
