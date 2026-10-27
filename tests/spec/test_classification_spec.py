"""One label per image, and the refusals that keep it from being read as a mask.

The task tag is not in `Task` yet, which is deliberate - it lands after the head and the
metrics, the way `0.8.0a20` landed the instance task - so nothing here goes through
`from_dict`. These are the two classes on their own, which is what the slice that builds
the head will be given.

What is worth asserting about a specification that nothing can yet validate is the part
that will not be obvious from the engine: **which configurations are refused.** Each one
below is a thing somebody would write by copying a segmentation spec, and each would train
if it were accepted - a `dice` loss on one label reports a plausible number, and `blocks: 6`
beside a backbone silently builds the backbone.
"""

from typing import get_args

import pytest
from pydantic import ValidationError

from pyplatypus.spec import ClassificationData, ClassificationModel, DataSpec
from pyplatypus.spec.components import LossSpec

#: The losses this package offers, read off the union rather than listed - the
#: argument `conftest.metric_configs` makes about metrics, and what
#: `tests/objectives/test_regression.py` already does with this same union.
LOSSES = get_args(get_args(LossSpec)[0])


def data(**over):
    """A valid classification data block. `validation: false` rather than a second path, so
    nothing here depends on a directory existing."""
    return ClassificationData(
        **{"train_path": ".", "validation": False, "classes": ["benign", "malignant"], **over}
    )


def model(**over):
    return ClassificationModel(**{"name": "stage", "input_shape": (64, 64), **over})


def refusal(call, *arguments, **keywords) -> str:
    """The message, with the assertion that there was one."""
    with pytest.raises(ValidationError) as caught:
        call(*arguments, **keywords)
    return str(caught.value)


def test_the_valid_block_is_valid():
    """The positive control, and it is not ceremony: every test below asserts a refusal, so
    a class that refused everything - a typo in a validator's condition - would satisfy all
    of them."""
    assert data().n_class == 2
    assert data().label_column == "label"
    assert model().loss.name == "cce"


def test_the_label_column_is_not_the_mask_column():
    """`DataSpec.label_column` answers `masks` and every task overrides it. Asserted
    because it is derived rather than declared, so nothing but this says it is right."""
    assert data().label_column == "label"
    assert DataSpec.label_column.fget(data()) == "masks"


def test_a_sample_is_an_image_so_there_are_no_subdirectories():
    """The field exists because `DataSpec` declares it, and this is what keeps
    `nested_dirs` from meaning two layouts at once."""
    message = refusal(data, subdirs=("images", "labels"))
    assert "classification sample is a single image" in message
    assert "one per class" in message or "directories under the path are the classes" in message
    assert data().subdirs is None


def test_one_class_is_not_a_classification():
    """Two at a minimum, where detection allows one. The message is pydantic's own, which is
    why this asserts the count rather than the wording."""
    assert "at least 2 items" in refusal(data, classes=["cell"])
    assert data(classes=["a", "b", "c"]).n_class == 3


def test_two_classes_with_one_name_are_refused():
    assert "must be distinct" in refusal(data, classes=["benign", "benign"])


def test_a_continuous_loss_over_a_label_is_refused_and_says_what_to_use():
    """`0.8.0a21`'s refusal, one task across: `loss` is the objective on the class
    channels. Every regression loss is covered because the refusal reads the flag."""
    continuous = [member.model_fields["name"].default for member in LOSSES if member.continuous]
    assert continuous, "read no continuous losses off the union; the extraction is the instrument"
    for name in continuous:
        message = refusal(model, loss={"name": name})
        assert "a label, not a number" in message
        assert "cce" in message and "focal" in message


def test_an_overlap_loss_is_refused_for_having_nothing_to_overlap():
    """The refusal this task needed that no other one has. Read off `needs_extent`, so the
    cases are every loss that declares itself defined over a region - including `lovasz`,
    which is extent-invariant in value and vacuous in mechanism."""
    over_a_region = [
        member.model_fields["name"].default
        for member in LOSSES
        if member.needs_extent and not member.continuous
    ]
    assert len(over_a_region) > 5, f"expected most of the losses here, got {over_a_region}"
    assert "lovasz" in over_a_region
    for name in over_a_region:
        message = refusal(model, loss={"name": name})
        assert "one label for a whole image" in message
        assert "Use cce, focal." in message


def test_the_admissible_losses_are_accepted():
    """The other half of the two refusals above, and the reason they are not one condition
    that refuses everything."""
    admissible = [
        member.model_fields["name"].default
        for member in LOSSES
        if not member.needs_extent and not member.continuous
    ]
    assert admissible == ["cce", "focal"], admissible
    for name in admissible:
        assert model(loss={"name": name}).loss.name == name


def test_our_knobs_beside_somebody_elses_backbone_are_refused():
    """Written, not merely different from the default: `model_fields_set` is the question.

    Proven able to fail by the second half - the same values arriving as defaults are
    accepted, which is what a condition reading the values instead would refuse.
    """
    for field, value in (("blocks", 6), ("filters", 32), ("block_width", 3)):
        message = refusal(model, encoder="resnet34", **{field: value})
        assert f"`{field}`" in message
        assert "a backbone replaces it entirely" in message

    built = model(encoder="resnet34")
    assert built.blocks == 4 and built.filters == 16 and built.block_width == 2


def test_a_side_that_runs_out_of_pixels_is_refused_but_an_odd_one_is_not():
    """Classification's half of segmentation's divisibility rule: nothing has to divide,
    and nothing may disappear."""
    message = refusal(model, blocks=6, input_shape=(48, 48))
    assert "at least 64" in message
    assert "no decoder to line up with" in message

    # 30 is not a multiple of 8 and survives three poolings: 15, 7, 3.
    assert model(blocks=3, input_shape=(30, 30)).blocks == 3
    assert model(blocks=5, input_shape=(32, 32)).blocks == 5


def test_the_architecture_is_the_contracting_path():
    """Derived rather than declared, which is the one place this task reads differently
    from its siblings - and the fingerprint has to follow it."""
    assert "architecture" not in ClassificationModel.model_fields
    assert model().architecture == "built_in"
    assert model(encoder="resnet34").architecture == "resnet34"

    ours = model().weights_fingerprint()
    theirs = model(encoder="resnet34").weights_fingerprint()
    assert ours["architecture"] == "built_in"
    assert theirs["architecture"] == "resnet34"
    # `blocks` and `filters` identify one of ours and shape nothing of theirs.
    assert {"blocks", "filters"} <= set(ours)
    assert not {"blocks", "filters"} & set(theirs)


def test_nothing_may_watch_a_number_this_task_does_not_report():
    """`foreground` is a segmentation epoch's number, and a callback watching it would
    otherwise be accepted and raise a `KeyError` after the first epoch - `0.8.0a27`'s
    failure, one task across."""
    assert model().monitorable == {"train_loss", "val_loss"}
    message = refusal(model, callbacks=[{"name": "early_stopping", "monitor": "val_foreground"}])
    assert "val_foreground" in message

    watching_the_loss = model(callbacks=[{"name": "early_stopping", "monitor": "val_loss"}])
    assert watching_the_loss.callbacks[0].monitor == "val_loss"


def test_a_metric_cannot_be_asked_for_yet():
    """The honest state rather than an oversight: `metrics` is the next slice, and
    segmentation's overlap metrics are not a menu this task can be offered. `extra=forbid`
    is what makes the absence a refusal instead of a silently ignored key."""
    assert "metrics" not in ClassificationModel.model_fields
    assert "Extra inputs are not permitted" in refusal(model, metrics=[{"name": "dice"}])


def test_the_transfer_settings_come_from_the_shared_base():
    """`EncoderModel`'s three refusals reach this task without being written twice, which
    is the whole reason that class exists. One assertion each, because the conditions
    themselves are tested where they live."""
    assert "pretrained=true needs `encoder`" in refusal(model, pretrained=True)
    assert "needs `encoder` to name a backbone" in refusal(model, freeze_encoder=2)
    assert "nothing to transfer to a volume" in refusal(
        model, encoder="resnet34", input_shape=(32, 64, 64)
    )
    # And the task's own rank is otherwise free: a volume classifier is ours, at rank 3.
    assert model(input_shape=(32, 64, 64)).rank == 3
