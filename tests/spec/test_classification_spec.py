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
from pyplatypus.spec.components import ClassificationMetricSpec, LossSpec

#: The losses this package offers, read off the union rather than listed - the
#: argument `conftest.metric_configs` makes about metrics, and what
#: `tests/objectives/test_regression.py` already does with this same union.
LOSSES = get_args(get_args(LossSpec)[0])

#: The same, for the metrics this task offers, and for the same reason.
METRICS = get_args(get_args(ClassificationMetricSpec)[0])


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
    assert model().architecture == "plain_cnn"
    assert model(encoder="resnet34").architecture == "resnet34"

    ours = model().weights_fingerprint()
    theirs = model(encoder="resnet34").weights_fingerprint()
    assert ours["architecture"] == "plain_cnn"
    assert theirs["architecture"] == "resnet34"
    # `blocks` and `filters` identify one of ours and shape nothing of theirs.
    assert {"blocks", "filters"} <= set(ours)
    assert not {"blocks", "filters"} & set(theirs)


def test_nothing_may_watch_a_number_this_task_does_not_report():
    """`foreground` is a segmentation epoch's number, and a callback watching it would
    otherwise be accepted and raise a `KeyError` after the first epoch - `0.8.0a27`'s
    failure, one task across.

    This asserted `monitorable == {"train_loss", "val_loss"}` until the metrics arrived,
    which is the sentence that had to move rather than a claim that was wrong: the set is
    the loss **plus what this model reports**, and the default now reports something.
    """
    assert {"train_loss", "val_loss"} <= model().monitorable
    assert not any("foreground" in name for name in model().monitorable)
    message = refusal(model, callbacks=[{"name": "early_stopping", "monitor": "val_foreground"}])
    assert "val_foreground" in message

    watching_the_loss = model(callbacks=[{"name": "early_stopping", "monitor": "val_loss"}])
    assert watching_the_loss.callbacks[0].monitor == "val_loss"


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


def test_the_default_metric_is_the_threshold_free_one():
    """`auc` rather than `accuracy`, which is the measurement in
    `tests/objectives/test_classification_metrics.py` arriving in the specification.

    Asserted by name rather than by count, because the defect this guards is a default that
    somebody made more conventional, and a list of one is as easy to change as to read.
    """
    assert [metric.name for metric in model().metrics] == ["auc"]


def test_every_metric_this_task_offers_can_be_asked_for_and_watched():
    """Read off the union, so a fifth metric is covered by arriving. The second assertion is
    the one that matters: a metric that can be configured and not watched is `0.8.0a27`."""
    for member in METRICS:
        name = member.model_fields["name"].default
        asked = model(metrics=[{"name": name}])
        assert [m.name for m in asked.metrics] == [name]
        assert {f"train_{name}", f"val_{name}"} <= asked.monitorable
        model(
            metrics=[{"name": name}],
            callbacks=[{"name": "early_stopping", "monitor": f"val_{name}"}],
        )


def test_the_overlap_metrics_are_not_offered_here():
    """A `dice` on one label is `f1` under a mask's name, and `smooth` and
    `include_background` are words about pixels. The union is the refusal."""
    with pytest.raises(ValidationError, match="dice"):
        model(metrics=[{"name": "dice"}])
    assert "cldice" not in {m.model_fields["name"].default for m in METRICS}


def test_a_callback_may_not_watch_a_metric_nobody_asked_for():
    """The epoch produces what the model reports and nothing else, so this is refused when
    the configuration is read rather than with a `KeyError` after the first epoch."""
    with pytest.raises(ValidationError, match="val_accuracy"):
        model(callbacks=[{"name": "early_stopping", "monitor": "val_accuracy"}])
    # and the same name is accepted the moment it is asked for
    model(
        metrics=[{"name": "accuracy"}],
        callbacks=[{"name": "early_stopping", "monitor": "val_accuracy"}],
    )


def test_there_is_no_worst_case_and_no_foreground_to_watch():
    """Both would be numbers this task's epochs do not produce.

    A segmentation epoch reports `val_<metric>_worst` because a case there is an image with
    its own score; one image's accuracy is 0 or 1, so a split's worst case is 0 for any
    model that gets one image wrong. `foreground` is the share of a prediction that is not
    background, and there is no background class here - which is what makes
    `terminate_on_collapse` refusable by name rather than at the end of an epoch.
    """
    watchable = model(metrics=[{"name": "accuracy"}, {"name": "auc"}]).monitorable
    assert watchable == {
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "train_auc",
        "val_auc",
    }
    assert not any(name.endswith("_worst") for name in watchable)
    with pytest.raises(ValidationError, match="val_foreground"):
        model(callbacks=[{"name": "terminate_on_collapse"}])


def test_the_metrics_are_part_of_what_a_weights_file_is_not_asked_about():
    """A fingerprint is what must match for a file of weights to belong to this model, and
    what was reported while training it is not that - two runs differing only in `metrics`
    produce interchangeable weights."""
    assert (
        model(metrics=[{"name": "auc"}]).weights_fingerprint()
        == model(metrics=[{"name": "accuracy"}, {"name": "f1"}]).weights_fingerprint()
    )
