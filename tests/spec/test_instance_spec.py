"""The third task: `instance_segmentation`, its two axes, and what they refuse.

Three things are asserted here and they are not the same thing.

The first is that an instance configuration is accepted and says what it means - the tag
selects `InstanceData` and `InstanceModel`, and the head's width comes out of the
separation and the data rather than out of a field somebody wrote.

The second is that **the two axes are a tree rather than free choices**. `follow_flows`
with `separation: {name: rim}` reads a signal that is not there; `loss: mse` scores classes
as if they were numbers; a flow block's `loss: dice` asks for the overlap between two
directions. Each of them would return a number and a gradient, which is the worst available
outcome, so each is refused at load time with both halves named. That is the same answer
`lovasz` with `regions` gets, and for the same reason.

**And one refusal is deliberately absent, which this file also asserts.** A mechanism's
settings live inside the block they belong to, so `width` with `flows` is an unknown key
rather than something a validator had to be written for - the flat form this replaced
needed a `_BELONGS_TO` table and six refusals to say what `extra="forbid"` now says.

The third is that **what was left unset stays unset**. `decode` and a flow block's `loss`
are derived when the model stayed silent, and the derivation is a property rather than a field
default: a specification has to keep being a description of what somebody wrote, both so
that `save_spec` round-trips and so that an older engine never receives a key it does not
know.
"""

import pytest

from pyplatypus import ConfigError, InstanceSpec, SegmentationSpec, Separation, Task, from_dict
from pyplatypus.engine import Engine, EngineError
from pyplatypus.spec import InstanceData, InstanceModel
from pyplatypus.spec.instances import (
    ConnectedComponents,
    Flows,
    FollowFlows,
    NearestSeed,
    Rim,
    Watershed,
)


@pytest.fixture
def instance_data():
    return {"train_path": ".", "validation_path": "."}


@pytest.fixture
def instance_config(instance_data):
    return {
        "task": "instance_segmentation",
        "data": instance_data,
        "models": [{"name": "nuclei", "input_shape": [256, 256], "separation": {"name": "rim"}}],
    }


def build(config):
    return from_dict(config, check_paths=False)


def with_model(config, **fields):
    """The same configuration with one model, overridden field by field."""
    return {**config, "models": [{**config["models"][0], **fields}]}


# --- the tag selects the shape ----------------------------------------------------------


def test_the_task_selects_the_instance_shape(instance_config):
    spec = build(instance_config)
    assert isinstance(spec, InstanceSpec)
    assert spec.task is Task.INSTANCE_SEGMENTATION
    assert isinstance(spec.data, InstanceData)
    assert isinstance(spec.models[0], InstanceModel)
    assert not isinstance(spec, SegmentationSpec)


def test_a_semantic_naming_is_refused_against_the_instance_task(instance_config):
    """`colormap` is how the *other* segmentation task names classes, and a directory of
    per-object files has nothing for it to name. Refused as an unexpected key against the
    task that was asked for, which is the whole point of the discriminator."""
    with pytest.raises(ConfigError) as caught:
        build({**instance_config, "data": {**instance_config["data"], "colormap": [[0, 0, 0]]}})
    assert "colormap" in str(caught.value)


def test_separation_is_refused_in_a_semantic_specification():
    with pytest.raises(ConfigError) as caught:
        from_dict(
            {
                "task": "semantic_segmentation",
                "data": {
                    "train_path": ".",
                    "validation_path": ".",
                    "colormap": [[0, 0, 0], [255, 255, 255]],
                },
                "models": [
                    {"name": "u", "input_shape": [256, 256], "separation": {"name": "rim"}},
                ],
            },
            check_paths=False,
        )
    assert "separation" in str(caught.value)


def test_separation_is_required_and_named_when_missing(instance_config):
    """No default, because it decides the head's width and neither value is a safe guess."""
    config = {**instance_config, "models": [{"name": "n", "input_shape": [256, 256]}]}
    with pytest.raises(ConfigError) as caught:
        build(config)
    assert "separation" in str(caught.value)


def test_the_data_names_two_classes_and_nothing_else_can_say_otherwise(instance_config):
    """`n_class` is derived, so there is no field a specification could disagree with."""
    spec = build(instance_config)
    assert spec.data.n_class == 2
    assert spec.n_class == 2
    assert "n_class" not in InstanceData.model_fields
    assert spec.data.multilabel is False
    assert spec.data.label_map is False


# --- separation is a block, and the block names a signal --------------------------------


def test_each_separation_block_names_its_signal():
    """`name` is how a block is spelled in a file and `signal` is what it means, and this
    is what holds the two together.

    A `ClassVar` cannot be checked by pydantic, so a block whose `signal` disagreed with
    its own `name` would build, validate and then build the wrong target - `rim`'s width
    through the flow builder, or a 3-channel head read as 4. The alternative was typing
    `name` as `Literal[Separation.RIM]`, which puts the enum into the generated schema for
    a reader who only ever writes a string; this is the cheaper half of that trade and the
    test is the other half.
    """
    blocks = {block.model_fields["name"].default: block for block in (Rim, Flows)}
    assert set(blocks) == {member.value for member in Separation}
    for name, block in blocks.items():
        assert block.signal.value == name, name
        assert block(name=name).signal is Separation(name)


def test_the_flat_form_is_not_a_separation(instance_config):
    """`separation: rim` was the first shape of this field and is not accepted now.

    Pinned rather than left to be discovered, because it is the one change in `#160` that a
    file written against an earlier reading of the issue would notice - and because the
    refusal has to arrive from the spec rather than from a target builder being handed a
    string it cannot use.
    """
    with pytest.raises(ConfigError) as caught:
        build(with_model(instance_config, separation="rim"))
    message = str(caught.value)
    assert "models[0].separation" in message
    assert "dictionary" in message


# --- the head's width, which is the piece that blocks everything else -------------------


@pytest.mark.parametrize(
    ("separation", "channels"),
    [("rim", 3), ("flows", 4)],
)
def test_the_head_width_is_derived_from_the_separation_and_the_data(
    instance_config, separation, channels
):
    """`rim` adds the third class, `flows` adds two continuous channels in front of the
    semantic block. This is the line `examples/train_instance_flows.py` had to write by
    hand as `build_model(spec, n_class=4)`, and the reason it lives outside `Engine`."""
    spec = build(with_model(instance_config, separation={"name": separation}))
    model = spec.models[0]
    assert model.head_channels(spec.data.n_class) == channels
    assert model.separation.head_channels(2) == channels


def test_the_class_block_starts_where_the_vector_channels_end(instance_config):
    """Everything that reads a mask out of the output needs this one number, and deriving
    it beside the width is what keeps the two from disagreeing.

    **The two targets put their extra channel in different places, which is the whole
    reason both numbers are derived and neither is a constant.** `flows` prefixes two
    continuous channels and leaves the classes as the data names them, so its class block
    is `n_class` wide and starts at 2. `rim` prefixes nothing and splits the foreground
    into interior and rim, so its block starts at 0 and is `n_class + 1` wide. A reader
    that assumed the class block was always the last `n_class` channels would be right
    about one of them and silently wrong about the other.
    """
    rim = build(with_model(instance_config, separation={"name": "rim"})).models[0]
    flows = build(with_model(instance_config, separation={"name": "flows"})).models[0]
    assert rim.separation.semantic_from == 0
    assert flows.separation.semantic_from == 2
    assert rim.head_channels(2) - rim.separation.semantic_from == 3
    assert flows.head_channels(2) - flows.separation.semantic_from == 2
    # Said once more as the invariant rather than as two numbers, so a third separation
    # cannot satisfy this by coincidence: the class block is the rest of the head, and the
    # one extra class belongs to the target that has no channels of its own.
    for model, extra in ((rim, 1), (flows, 0)):
        assert model.head_channels(2) == model.separation.semantic_from + 2 + extra


def test_hv_is_not_a_separation_value(instance_config):
    """Measured rather than unimplemented, and the absence is the result.

    `pyplatypus#213` swept both ways HoVer-Net's distance maps can be turned into markers,
    on the ground truth: 0.9382 and 0.7995 against a one-pixel erosion's 0.9807. The better
    recipe loses to the cheapest marker there is on perfect inputs, so a trained one cannot
    win, and naming the value would promise a mechanism this package measured as absent.
    """
    assert {member.value for member in Separation} == {"rim", "flows"}
    with pytest.raises(ConfigError) as caught:
        build(with_model(instance_config, separation={"name": "hv"}))
    assert "hv" in str(caught.value)


# --- decode: derived when silent, a list when stated ------------------------------------


def test_the_decodings_are_the_signal_and_the_yardstick_when_unset(instance_config):
    """The pair that makes a row readable: the same model and mask with and without the
    separation signal. `connected_components` is second because it is what the first is
    read against."""
    rim = build(with_model(instance_config, separation={"name": "rim"})).models[0]
    flows = build(with_model(instance_config, separation={"name": "flows"})).models[0]
    assert [type(block) for block in rim.decodings] == [NearestSeed, ConnectedComponents]
    assert [type(block) for block in flows.decodings] == [FollowFlows, ConnectedComponents]


def test_an_unset_decode_stays_unset_in_what_is_sent_and_saved(instance_config):
    """Derived as a property, never written into the specification.

    Two things rest on this: an older engine receiving R's request never meets a key it
    does not know, and `model_fields_set` can still tell a stated single decoding from the
    pair that was there anyway.
    """
    spec = build(instance_config)
    model = spec.models[0]
    assert model.decode is None
    assert "decode" not in model.model_fields_set
    assert spec.to_dict()["models"][0]["decode"] is None
    # And the derivation is not empty, or the assertion above would pass on a model that
    # decodes nothing at all.
    assert len(model.decodings) == 2


def test_a_stated_decode_is_exactly_what_was_stated(instance_config):
    spec = build(
        with_model(
            instance_config,
            separation={"name": "rim"},
            decode=[
                {"name": "watershed", "minimum_area": 15},
                {"name": "connected_components"},
            ],
        )
    )
    model = spec.models[0]
    assert [block.name for block in model.decodings] == ["watershed", "connected_components"]
    assert model.decodings[0].minimum_area == 15
    assert "decode" in model.model_fields_set


def test_an_empty_decode_list_is_refused(instance_config):
    """A model that decodes nothing produces no objects and no score, which is not a
    configuration anybody means - and it is the shape that would make every other
    assertion here pass over an empty list."""
    with pytest.raises(ConfigError):
        build(with_model(instance_config, decode=[]))


def test_a_misspelled_decoding_fails_by_name(instance_config):
    with pytest.raises(ConfigError) as caught:
        build(with_model(instance_config, decode=[{"name": "watershad"}]))
    assert "watershad" in str(caught.value)


def test_a_parameter_the_decoding_does_not_have_is_refused(instance_config):
    """The same promise every other block here makes: `{"name": "watershed", "steps": 3}`
    fails rather than silently doing nothing."""
    with pytest.raises(ConfigError) as caught:
        build(with_model(instance_config, decode=[{"name": "watershed", "steps": 3}]))
    assert "steps" in str(caught.value)


# --- the tree: a decoding needs the signal it reads -------------------------------------


@pytest.mark.parametrize(
    ("separation", "decoding", "wanted"),
    [
        ("rim", "follow_flows", "flows"),
        ("flows", "nearest_seed", "rim"),
        ("flows", "watershed", "rim"),
    ],
)
def test_a_decoding_that_reads_another_signal_is_refused_naming_both(
    instance_config, separation, decoding, wanted
):
    with pytest.raises(ConfigError) as caught:
        build(
            with_model(
                instance_config, separation={"name": separation}, decode=[{"name": decoding}]
            )
        )
    message = str(caught.value)
    assert decoding in message
    assert separation in message
    assert wanted in message


@pytest.mark.parametrize("separation", ["rim", "flows"])
def test_connected_components_is_admissible_with_every_target(instance_config, separation):
    """It reads none of the signals, which is why it is the yardstick rather than slack."""
    spec = build(
        with_model(
            instance_config,
            separation={"name": separation},
            decode=[{"name": "connected_components"}],
        )
    )
    assert [block.name for block in spec.models[0].decodings] == ["connected_components"]


@pytest.mark.parametrize(
    ("separation", "decoding"),
    [("rim", "nearest_seed"), ("rim", "watershed"), ("flows", "follow_flows")],
)
def test_the_signals_own_decodings_are_admissible(instance_config, separation, decoding):
    """The other half of the refusal above: a check that only ever said no would pass by
    refusing everything."""
    spec = build(
        with_model(instance_config, separation={"name": separation}, decode=[{"name": decoding}])
    )
    assert spec.models[0].decodings[0].name == decoding


def test_which_signal_a_decoding_needs_is_read_off_the_decoding(instance_config):
    """A class variable beside the algorithm's own description, not a table in a validator
    - so a new decoding declares its requirement where somebody adding one will see it."""
    assert ConnectedComponents.needs == frozenset()
    assert NearestSeed.needs == frozenset({Separation.RIM})
    assert Watershed.needs == frozenset({Separation.RIM})
    assert FollowFlows.needs == frozenset({Separation.FLOWS})


# --- the tree: the mechanism decides which losses are admissible ------------------------


@pytest.mark.parametrize("separation", ["rim", "flows"])
def test_a_continuous_loss_is_refused_on_the_semantic_channels(instance_config, separation):
    """`loss` is the objective on the mask at either separation. Run as it stands, `mse`
    against a one-hot target scores raw output against 0 and 1 with no activation between
    them, and returns a number for it."""
    with pytest.raises(ConfigError) as caught:
        build(with_model(instance_config, separation={"name": separation}, loss={"name": "mse"}))
    message = str(caught.value)
    assert "mse" in message
    assert "separation: flows" in message


def test_an_overlap_loss_is_refused_on_the_vector_channels(instance_config):
    with pytest.raises(ConfigError) as caught:
        build(with_model(instance_config, separation={"name": "flows", "loss": {"name": "dice"}}))
    message = str(caught.value)
    assert "dice" in message
    assert "mse" in message


def test_the_vector_objective_is_mse_on_the_foreground_when_unset(instance_config):
    """What the measured run used, and `where` is the field to read before changing it:
    the foreground is 13.97% of a Data Science Bowl image."""
    model = build(with_model(instance_config, separation={"name": "flows"})).models[0]
    assert model.separation.loss is None
    assert model.objective.name == "mse"
    assert model.objective.where == "foreground"


def test_a_stated_vector_objective_is_the_one_used(instance_config):
    model = build(
        with_model(
            instance_config,
            separation={
                "name": "flows",
                "loss": {"name": "smooth_l1", "beta": 0.1},
                "weight": 2.0,
            },
        )
    ).models[0]
    assert model.objective.name == "smooth_l1"
    assert model.objective.beta == 0.1
    assert model.separation.weight == 2.0


@pytest.mark.parametrize(
    ("separation", "field", "value"),
    [
        ("rim", "loss", {"name": "mse"}),
        ("rim", "weight", 2.0),
        ("rim", "mode", "centroid"),
        ("flows", "width", 2),
    ],
)
def test_a_mechanism_setting_is_unwritable_outside_its_mechanism(
    instance_config, separation, field, value
):
    """**Unwritable rather than refused**, which is the block form's whole argument.

    The flat form put these four beside `separation` as `rim_width`, `flow_mode`,
    `flow_loss` and `flow_weight`, where every one of them parses on every model - so it
    needed a `_BELONGS_TO` table and a validator to refuse the half that means nothing, and
    a fifth setting would have had to be added to both. Here a setting is a key of the block
    it belongs to, so the wrong crossing is an unknown key and `extra="forbid"` answers it
    with the message every typo in a specification gets.

    The refusal is still asserted rather than assumed, and it still names the field and the
    block it was written under - which is what a reader needs in order to see which of the
    two they got wrong.
    """
    with pytest.raises(ConfigError) as caught:
        build(with_model(instance_config, separation={"name": separation, field: value}))
    message = str(caught.value)
    assert field in message
    assert separation in message
    assert "Extra inputs are not permitted" in message


@pytest.mark.parametrize(
    ("separation", "field", "value", "settings"),
    [
        ("rim", "width", 2, {"width": 2}),
        ("flows", "mode", "centroid", {"mode": "centroid"}),
    ],
)
def test_a_mechanism_setting_is_accepted_inside_its_mechanism(
    instance_config, separation, field, value, settings
):
    """The other half of the refusal above: a check that only ever said no would pass by
    refusing every combination, including the two that are the point."""
    model = build(
        with_model(instance_config, separation={"name": separation, field: value})
    ).models[0]
    assert getattr(model.separation, field) == value
    assert model.target_settings == settings


def test_the_defaults_are_the_ones_their_documentation_argues_for(instance_config):
    """Both defaults are pinned here, because a default that drifts away from the number
    its own documentation quotes is the §4 defect this project writes tests against.

    `diffusion` is a measurement: 0.9924 decoded against `centroid`'s 0.9478. The rim width
    is **not**, and the field says so - the published sweep peaked at 3 pixels of *source*
    resolution, which on a 520-pixel case read at 256 is about 1.5 pixels of model input,
    and this field is in model pixels. 2 is the nearest width above that and a sweep in
    these units has not been run. The test pins the two numbers to the two arguments, which
    is all a test can do about a default nobody has measured yet.
    """
    rim = build(with_model(instance_config, separation={"name": "rim"})).models[0]
    flows = build(with_model(instance_config, separation={"name": "flows"})).models[0]
    assert rim.separation.width == 2
    assert flows.separation.mode == "diffusion"
    assert rim.target_settings == {"width": 2}
    assert flows.target_settings == {"mode": "diffusion"}
    # Neither is in the block's `model_fields_set`, so both travel as defaults rather than
    # as settings somebody wrote - which is what `save_spec` has to round-trip.
    assert "width" not in rim.separation.model_fields_set
    assert "mode" not in flows.separation.model_fields_set


def test_a_mode_the_flow_builder_does_not_have_is_refused(instance_config):
    """The two modes are the two `FLOW_MODES` the builder accepts, and the spec refuses a
    third before the first sample is read rather than letting `flows_from_instances` raise
    on it."""
    from pyplatypus.instances import FLOW_MODES

    assert set(FLOW_MODES) == {"centroid", "diffusion"}
    with pytest.raises(ConfigError) as caught:
        build(with_model(instance_config, separation={"name": "flows", "mode": "sinkhorn"}))
    assert "sinkhorn" in str(caught.value)


def test_rim_takes_the_ordinary_softmax_losses(instance_config):
    """Three mutually exclusive classes, so this needs nothing new: it is the same loss the
    published `dsbowl-unet` weights were trained with, over one more class."""
    model = build(
        with_model(instance_config, separation={"name": "rim"}, loss={"name": "cce_dice"})
    ).models[0]
    assert model.loss.name == "cce_dice"
    assert model.objective is None


def test_the_admissibility_is_read_off_the_loss_rather_than_from_a_list(instance_config):
    """A new regression loss has to be refused on the semantic channels without this file
    or the instance spec being edited, which is what the flag on the loss is for."""
    from pyplatypus.spec.components import CceLoss, MseLoss

    assert MseLoss().continuous is True
    assert CceLoss().continuous is False


# --- the rest of a model is the rest of a model -----------------------------------------


def test_instances_are_two_dimensional(instance_config):
    """Both targets and all four decodings are written for images, and the readers refuse
    a 3-D label map by name. This refuses the specification that would reach them."""
    with pytest.raises(ConfigError) as caught:
        build(with_model(instance_config, input_shape=[64, 64, 64]))
    message = str(caught.value)
    assert "2-D" in message or "2D" in message


def test_the_shared_validators_still_apply(instance_config):
    """`InstanceModel` is a `SegmentationModel`, so the things that were already true of a
    model are true here - and these two are the ones that live on the spec rather than on
    the model, so inheritance alone does not establish them."""
    two = {**instance_config, "models": [instance_config["models"][0]] * 2}
    with pytest.raises(ConfigError, match="unique"):
        build(two)
    with pytest.raises(ConfigError, match="divisible"):
        build(with_model(instance_config, input_shape=[250, 250]))


def test_a_callback_cannot_watch_a_number_no_split_will_produce(instance_data):
    """The validator shared with the semantic task, reached through the instance one - the
    thing that went wrong twice in `EngineBase` was two tasks answering one question
    differently, so the answer has one home and both tasks are tested against it."""
    with pytest.raises(ConfigError, match="val_"):
        from_dict(
            {
                "task": "instance_segmentation",
                "data": {"train_path": ".", "validation": False},
                "models": [
                    {
                        "name": "n",
                        "input_shape": [256, 256],
                        "separation": {"name": "rim"},
                        "callbacks": [{"name": "early_stopping", "monitor": "val_loss"}],
                    }
                ],
            },
            check_paths=False,
        )


def test_the_weights_fingerprint_tells_a_rim_head_from_a_flows_head(instance_config):
    """At `n_class` 2 a `rim` head is 3 channels and a `flows` head is 4, but the point is
    not the shape: two heads of the same width and architecture that mean entirely
    different things would otherwise load into each other and predict confidently wrong.
    """
    rim = build(with_model(instance_config, separation={"name": "rim"})).models[0]
    flows = build(with_model(instance_config, separation={"name": "flows"})).models[0]
    assert rim.weights_fingerprint()["separation"] == "rim"
    assert flows.weights_fingerprint()["separation"] == "flows"
    assert rim.weights_fingerprint() != flows.weights_fingerprint()
    # And everything the semantic fingerprint already carried is still in it.
    assert {"architecture", "input_shape", "channels", "rank", "blocks", "filters"} <= set(
        rim.weights_fingerprint()
    )


# --- which engine trains it ---------------------------------------------------------


def test_the_semantic_engine_sends_an_instance_spec_to_the_right_engine(instance_config):
    """This test asserted that nothing trained instances yet, and then something did.

    Kept rather than deleted, pointing the other way: the message a reader gets when they
    hand an instance specification to `Engine` is the only thing that tells them which
    engine to reach for, and a message naming the task without naming the engine is the
    dead end this is here to prevent. `build_engine` is tested where it can reach data,
    in `tests/test_instance_engine.py`.
    """
    spec = build(instance_config)
    with pytest.raises(EngineError) as caught:
        Engine(spec)
    message = str(caught.value)
    assert "instance_segmentation" in message
    assert "InstanceEngine" in message
    # Not the detection message, which is the mistake a single "not segmentation" branch
    # would have made - and did, until the task was named.
    assert "detect_blood_cells" not in message
