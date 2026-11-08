"""What a reported figure depends on, beyond the weights and the data.

A weights sidecar recorded `trained_with` and the figures it measured, and **not what the
figures were measured with**. For `dice` that is harmless: it takes `include_background` and
nothing else. For `cldice` it is not - `iterations` is the skeleton's peel count, it moves
the published `fives-unet` number by **0.049**, and it was written nowhere at all, because it
arrived as `ClDice.__init__`'s default through a publisher's own flag. So *is this figure
reproducible* had to be answered by sweeping the parameter until a run landed on the
published digits. `pyplatypus#413` is that day; `#419` is the key that makes it a reading.

`measurement_fingerprint()` is asked of the spec for the reason `weights_fingerprint`'s own
docstring gives one method up: **the answer is task-specific.** Segmentation reports a
configured list of metrics; a detector has no `metrics` field at all and reports one table
whose rows move with three thresholds. A fixed list in `pyplatypus.weights` would have to
grow a branch per task and reach for a field that is not there.

What this file holds is the *class* rather than the four instances of it. Every member of
`AnySpec` is asked, by introspection, so a task added next month either answers or is named
here with a reason - which is the failure this method was written against, since a new task
would otherwise inherit the base class's silence and nothing would say so.
"""

from __future__ import annotations

import json
from typing import get_args

import pytest

from pyplatypus import from_dict
from pyplatypus.spec.models import ModelSpec
from pyplatypus.spec.spec import AnySpec

#: The model class of every task, read off the union rather than listed - `test_task_coverage`
#: does the same and for the same reason: a list maintained by hand is a list that will be
#: wrong, and a task's tag lands last.
MODEL_CLASSES = {
    spec.__name__: get_args(spec.model_fields["models"].annotation)[0]
    for spec in get_args(get_args(AnySpec)[0])
}

#: A task whose figures nothing configurable decides, with the reason. **It is empty**, and
#: that is the state to notice rather than a gap: all four tasks have something, and the
#: entry this list would hold is a task reporting one fixed table with no thresholds and no
#: metric list. An empty exemption list needs the control below, or `assert` over it is an
#: assertion about nothing.
NOTHING_DECIDES_THEIR_FIGURES: dict[str, str] = {}


def test_the_model_classes_were_found_at_all():
    """Or every parametrised case below is a case over an empty mapping."""
    assert len(MODEL_CLASSES) == 4, (
        f"found {sorted(MODEL_CLASSES)} - `AnySpec`'s shape changed, and this file's "
        "introspection has to follow it rather than quietly covering fewer tasks"
    )


def test_the_base_class_answers_EMPTY_so_a_new_task_cannot_inherit_a_wrong_answer():
    """The base is silent on purpose, and the next test is what makes silence loud.

    A base returning something plausible - the metrics, say - would answer for a task that
    has no metrics and read as a record rather than as an omission. Empty plus an
    enumeration is the pair: the method cannot invent, and nothing can stay quiet.
    """
    assert ModelSpec.measurement_fingerprint(object.__new__(ModelSpec)) == {}


@pytest.mark.parametrize("task", sorted(MODEL_CLASSES), ids=sorted(MODEL_CLASSES))
def test_every_task_says_what_its_figures_were_measured_with(task):
    """The enumeration. A task that answers nothing is named above or fails here."""
    model = MODEL_CLASSES[task]
    assert "measurement_fingerprint" in vars(model) or task in NOTHING_DECIDES_THEIR_FIGURES, (
        f"{model.__name__} does not override `measurement_fingerprint`, so it inherits the "
        f"base class's empty answer - which reads as *nothing configurable decides this "
        f"task's figures*. Either that is true and {task} belongs in "
        "NOTHING_DECIDES_THEIR_FIGURES with the reason, or the method is owed"
    )


@pytest.mark.parametrize("task", sorted(MODEL_CLASSES), ids=sorted(MODEL_CLASSES))
def test_what_a_task_names_are_fields_it_actually_has(task):
    """A fingerprint naming a field that no longer exists is a record of nothing.

    Checked against the model's own fields rather than against a list, so a rename fails
    here instead of writing `null` into every sidecar published afterwards. `metrics` is
    the one key that is not a field name - it is the dump of one - and is allowed by name.
    """
    model = MODEL_CLASSES[task]
    if "measurement_fingerprint" not in vars(model):
        pytest.skip(f"{task} is exempt and the test above is what holds that")
    source = inspect_keys(model)
    assert source, f"could not read {model.__name__}'s fingerprint keys"
    for key in source:
        assert key in model.model_fields or key == "metrics", (
            f"{model.__name__}.measurement_fingerprint() records '{key}', which is not one "
            f"of its fields. A renamed field would otherwise write null into every sidecar"
        )


def inspect_keys(model) -> set[str]:
    """The literal keys one class's own `measurement_fingerprint` writes, by AST.

    Its own override only - `super()` is what carries the inherited ones, and those are the
    parent's to justify.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(model.measurement_fingerprint)))
    return {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


# --------------------------------------------------------------------------------------
# And the behaviour, on the case this exists for.


def _segmentation(metrics, **model):
    return from_dict(
        {
            "task": "semantic_segmentation",
            "data": {
                "train_path": "a",
                "validation_path": "b",
                "colormap": [[0, 0, 0], [255, 255, 255]],
            },
            "models": [{"name": "m", "input_shape": [32, 32], "metrics": metrics, **model}],
        },
        check_paths=False,
    ).models[0]


def _data(**fields):
    """The same configuration's DATA half, which is where `pyplatypus#447`'s fields live.

    `validation_path=None` drops the key, so a caller can hand it a `split` instead - the
    two are exclusive and the validator says so.
    """
    fields = {k: v for k, v in fields.items() if v is not None}
    return from_dict(
        {
            "task": "semantic_segmentation",
            "data": {
                **({} if "split" in fields else {"validation_path": "b"}),
                "train_path": "a",
                "colormap": [[0, 0, 0], [255, 255, 255]],
                **fields,
            },
            "models": [{"name": "m", "input_shape": [32, 32]}],
        },
        check_paths=False,
    ).data


def test_a_metric_parameter_NOBODY_WROTE_is_recorded():
    """The defect's own case, and the whole argument for dumping rather than copying.

    `iterations` is not in this configuration. It is `ClDice`'s default, and it is the value
    that made `#413` a day of measurement - so a fingerprint that recorded the configuration
    as written would record nothing here, which is exactly the state this replaces.
    """
    model = _segmentation([{"name": "cldice"}])
    recorded = model.measurement_fingerprint()["metrics"]

    assert len(recorded) == 1
    assert recorded[0]["name"] == "cldice"
    assert recorded[0]["iterations"] == 5, (
        "the peel count is missing or wrong. It is the parameter whose value moved the "
        "published clDice by 0.049 and appeared in no file"
    )
    # The control that this is a resolved dump and not a passthrough: nothing above wrote
    # `iterations`, and `smooth` is a second default arriving the same way.
    assert "smooth" in recorded[0]


def test_a_value_somebody_DID_write_is_recorded_as_written():
    """The other direction, or the test above passes on a function returning a constant."""
    model = _segmentation([{"name": "cldice", "iterations": 12}])
    assert model.measurement_fingerprint()["metrics"][0]["iterations"] == 12


def test_the_detection_fingerprint_names_the_thresholds_that_move_its_table(detection_config):
    """A detector has no `metrics`, and an empty record would have read as *nothing to say*."""
    model = from_dict(detection_config, check_paths=False).models[0]
    recorded = model.measurement_fingerprint()

    assert set(recorded) == {"score_threshold", "nms_threshold", "operating_point"}, (
        f"detection records {sorted(recorded)}. All three change a published row while the "
        "weights and the data stay the same, which is the test for belonging here"
    )
    assert "metrics" not in recorded, (
        "a detector has no metric list, and writing an empty one would say that nothing "
        "configurable decides its figures"
    )
    # `ignore_threshold` is a training parameter: it shapes the target's ignore mask, so it
    # changes what the weights learnt rather than how they are scored. Asserted so that
    # adding it later is a decision rather than a slip.
    assert "ignore_threshold" in type(model).model_fields
    assert "ignore_threshold" not in recorded


def test_the_TILING_is_recorded_because_it_decides_which_pictures_were_scored():
    """`pyplatypus#427`: the field that was out and is one.

    A source image is read at `splits * input_shape` and cut, so the tiling changes the
    pictures the metric sees rather than the model. On the published `fives-unet` - trained
    and scored at four by four - a re-score at two by two reads Dice 0.8054393 against
    0.8788833, and `weights_fingerprint` cannot refuse it: only `input_shape` is checked and
    every tile is `input_shape` either way. So the sidecar for those two runs was identical
    and one of the two numbers was wrong for it.

    Both directions, because recording a constant would satisfy either one alone.
    """
    untiled = _segmentation([{"name": "dice"}])
    tiled = _segmentation([{"name": "dice"}], splits=[4, 4])

    assert untiled.measurement_fingerprint()["splits"] is None, (
        "a model that resizes instead of tiling has to say so rather than leave the key out: "
        "absent and `null` read alike to `jsonlite`, which is `0.8.0a74`'s own lesson"
    )
    assert tiled.measurement_fingerprint()["splits"] == [4, 4]
    assert tiled.measurement_fingerprint() != untiled.measurement_fingerprint(), (
        "two runs that score different pictures write the same sidecar, which is the whole "
        "of what this records"
    )


def test_an_instance_model_inherits_the_tiling_as_well_as_its_own_two():
    """The other half of `#427`: an instance figure is per decoding AND per tiling.

    `InstanceModel` extends `SegmentationModel`, so this is inheritance rather than a second
    copy - and it is asserted because nothing else says the inheritance is wanted, which is
    how `surface_dice` reached one side of the whole-mask pair and not the other.
    """
    model = from_dict(
        {
            "task": "instance_segmentation",
            "data": {"train_path": ".", "validation_path": "."},
            "models": [
                {
                    "name": "nuclei",
                    "input_shape": [256, 256],
                    "separation": {"name": "rim"},
                    "splits": [2, 2],
                }
            ],
        },
        check_paths=False,
    ).models[0]
    recorded = model.measurement_fingerprint()

    assert "splits" in recorded, (
        f"an instance model records {sorted(recorded)}. It can tile exactly as a semantic "
        "one can, and the field it inherits is the one that decides what was scored"
    )
    assert recorded["splits"] == [2, 2]
    assert {"separation", "decode", "metrics"} <= set(recorded)


# ------------------------------------------------------- the data's half, `pyplatypus#447`
#
# `#427` recorded `splits` because a source image is read at `splits * input_shape` and cut,
# so the tiling changes the pictures a metric sees while the weights and the data stay the
# same. Three fields one layer out pass that same test and are the DATA's: `window`,
# `channels_from` and `target_spacing`. They cannot live on the model's method, whose first
# sentence is *beyond the weights and the data*, and `pyplatypus.weights` is handed a model
# and never sees a data spec - so the division is one method per half and the engine joins
# them, exactly as it already does for `_class_fingerprint`.

#: The data class of every task, read off the union for `MODEL_CLASSES`' reason rather than
#: listed. A task's data class is where a field that shapes what gets read actually lands.
DATA_CLASSES = {
    spec.__name__: spec.model_fields["data"].annotation for spec in get_args(get_args(AnySpec)[0])
}


def test_the_data_classes_were_found_at_all():
    """Or every parametrised case below is a case over an empty mapping."""
    assert len(DATA_CLASSES) == 4, (
        f"found {sorted(DATA_CLASSES)} - `AnySpec`'s shape changed and this file's "
        "introspection has to follow it rather than quietly covering fewer tasks"
    )


@pytest.mark.parametrize("task", sorted(DATA_CLASSES))
def test_every_tasks_data_says_how_it_was_READ(task):
    """`window` is on the base, so every task answers rather than three of four.

    It is the one of the three that is not segmentation's alone, and it is also the one that
    applies to a format every task can read. A method on `SegmentationData` only would have
    left a CT classifier's sidecar silent about the windowing its figures were measured
    under, which is the shape `surface_dice` had when it reached one side of the whole-mask
    pair and not the other.
    """
    fingerprint = DATA_CLASSES[task].measurement_fingerprint
    assert fingerprint is not None
    # Asked of an instance rather than of the class, because the answer is the resolved
    # value and a class cannot have one.
    data = DATA_CLASSES[task].model_construct(window="auto")
    assert data.measurement_fingerprint() == {"window": "auto"}


def test_the_window_is_recorded_EVEN_WHEN_NOBODY_WROTE_ONE():
    """`'auto'` is a windowing rather than an absence, which is why it is not skipped.

    This is the resolved-rather-than-as-written rule the model's half gives in full: the
    value that made `#413` a day of measurement was a default nobody typed. A sidecar that
    recorded `window` only when a configuration named it would be silent in exactly the case
    where a reader cannot guess - and `auto` means *the window the file recorded*, so two
    datasets read with `auto` are not necessarily read alike.
    """
    assert _data().measurement_fingerprint() == {"window": "auto"}


def test_the_two_segmentation_ONLY_fields_are_recorded_only_when_SET():
    """The author's decision of 2026-11-08, and both directions of it.

    Three `null`s would say *measured without a window* where the truth is *written by an
    engine that could not answer*, and those are different facts - the same distinction
    `0.8.0a74` drew in R, where `jsonlite` returns `NULL` for a JSON `null` and for a key
    that is not there alike. Absence is the honest answer for a field nobody set, because
    `_class_fingerprint` beside it already works that way: it writes the one naming that is
    there.
    """
    unset = _data().measurement_fingerprint()
    assert set(unset) == {"window"}, (
        f"a run that named neither field records {sorted(unset)}; the two it did not name "
        "are absent rather than null"
    )

    named = _data(
        channels_from=["*_t1.nii", "*_t2.nii"], target_spacing=[1.0, 1.0, 3.0]
    ).measurement_fingerprint()
    assert named["channels_from"] == ["*_t1.nii", "*_t2.nii"]
    assert named["target_spacing"] == [1.0, 1.0, 3.0]
    assert named != unset, (
        "two runs that read the same files into different volumes write the same record, "
        "which is the whole of what this adds"
    )


def test_each_of_the_two_is_recorded_on_its_OWN():
    """Or the test above passes on a method that records both or neither.

    `channels_from` 0.467 against 0.110 of Dice on one trained model, and `target_spacing`
    18432 real voxels against 4608 at an identical model input shape - two independent
    facts, so a run naming one of them has to say which.
    """
    channels = _data(channels_from=["*_t1.nii", "*_t2.nii"]).measurement_fingerprint()
    spacing = _data(target_spacing=[2.0, 2.0, 2.0]).measurement_fingerprint()

    assert set(channels) == {"window", "channels_from"}
    assert set(spacing) == {"window", "target_spacing"}


def test_an_explicit_window_is_a_PAIR_a_json_file_can_hold():
    """A named window is a string and an explicit one is a `(centre, width)` tuple.

    Converted here rather than left to whoever serialises, so that what this returns is
    already what the file will hold: `json.dumps` writes a tuple as a list and
    `safetensors`' metadata would not take it at all, and a record that survives one
    serialiser and not another is a record whose shape depends on its reader.
    """
    recorded = _data(window=[40.0, 400.0]).measurement_fingerprint()
    assert recorded["window"] == [40.0, 400.0]
    assert isinstance(recorded["window"], list)
    assert json.loads(json.dumps(recorded))["window"] == [40.0, 400.0]


def test_the_datas_half_and_the_models_half_share_no_key_TODAY():
    """The guard in `export_weights` refuses a clash; this says there is not one yet.

    Both halves land in one `measured_with` block, so a `window` arriving as a model field
    some day would be two answers to one question. That is a refusal rather than a silent
    overwrite - and this is the control that the refusal is not currently firing, which a
    test of the refusal alone cannot tell you.
    """
    # Two patterns, because `channels_from` names the files that BECOME the channels and
    # one of them is not a channel ordering at all - the validator says so.
    data = _data(channels_from=["*_t1.nii", "*_t2.nii"], target_spacing=[1.0, 1.0, 1.0])
    model = _segmentation([{"name": "cldice"}], splits=[2, 2])

    shared = set(data.measurement_fingerprint()) & set(model.measurement_fingerprint())
    assert shared == set(), f"the two halves both claim {sorted(shared)}"


# ------------------------------------------------ WHICH cases were scored, `pyplatypus#450`
#
# `#447` recorded how the data was READ and nothing about which cases the figures came from.
# That issue named three candidates and the measurement that scoped it says ONE belongs,
# which is the same shape `#447` had one issue earlier, where it named four fields and
# `labels` turned out to be recorded already:
#
#     split        IN.  `fractions`, `group_by` and `seed` with the discovered samples decide
#                  the partition, so two runs on one folder at different fractions score
#                  different pictures. It changes a NUMBER.
#     shuffle      OUT. Training only, by its own description and by every call site: each
#                  `shuffle=self.spec.data.shuffle` is on the "train" split and every scoring
#                  and predicting loader passes `shuffle=False`. It changes what the weights
#                  LEARNT, which is `ignore_threshold`'s class in the detection test above.
#     validation   OUT. It gates whether a validation split EXISTS - `false` makes asking for
#                  one a refusal by name - rather than changing the value of a figure.


def test_the_SPLIT_is_recorded_because_it_decides_which_cases_were_scored():
    """`pyplatypus#450`. Both directions, and the absent one is a real case.

    `CLAUDE.md` §4 has what the silence costs: `dsbowl-unet`'s card reports Dice over "134
    held-out images", the 134 case directories on this disk are a **different** 134, and the
    discrepancy that gave it away was 0.01 of Dice - which is also what a rounding difference
    looks like.
    """
    partitioned = _data(
        train_path="a", validation_path=None, split={"fractions": [0.8, 0.2], "group_by": None}
    )
    named = _data()

    assert "split" not in named.measurement_fingerprint(), (
        "a run whose second folder is named partitioned nothing, and the record says so by "
        "saying nothing - `only what is set`, which has a real second case here"
    )
    recorded = partitioned.measurement_fingerprint()["split"]
    assert recorded["fractions"] == [0.8, 0.2]
    assert partitioned.measurement_fingerprint() != named.measurement_fingerprint()


def test_the_splits_SEED_arrives_resolved_although_nobody_wrote_it():
    """The `#413` argument one field over, and the half that makes the record reproducible.

    `fractions` alone describes a partition; `fractions` plus `group_by` plus `seed` plus the
    discovered samples **determines** it. The seed defaults to 0 and arrives by resolution, so
    a record of the configuration as written would carry a partition nobody can rebuild.
    """
    default = _data(
        train_path="a", validation_path=None, split={"fractions": [0.8, 0.2], "group_by": None}
    )
    eleven = _data(
        train_path="a",
        validation_path=None,
        split={"fractions": [0.8, 0.2], "group_by": None, "seed": 11},
    )

    assert default.measurement_fingerprint()["split"]["seed"] == 0, (
        "the seed is missing or unresolved; nothing above wrote it, which is exactly the case "
        "this records"
    )
    assert eleven.measurement_fingerprint()["split"]["seed"] == 11
    assert (
        default.measurement_fingerprint()["split"] != eleven.measurement_fingerprint()["split"]
    ), "two runs that hold out different pictures write the same record"


def test_the_splits_GROUP_BY_is_recorded_because_it_is_the_leak_the_examples_exist_for():
    """`group_by` is what makes a split by patient rather than by slice.

    `examples/measure_split_leak.py` exists for the number this field moves, so a sidecar
    that recorded `fractions` and not `group_by` would record the shape of a partition and
    not the thing that makes it honest.
    """
    by_slice = _data(
        train_path="a", validation_path=None, split={"fractions": [0.8, 0.2], "group_by": None}
    )
    by_case = _data(
        train_path="a",
        validation_path=None,
        split={"fractions": [0.8, 0.2], "group_by": "^(case[0-9]+)_"},
    )

    assert by_case.measurement_fingerprint()["split"]["group_by"] == "^(case[0-9]+)_"
    assert by_slice.measurement_fingerprint()["split"]["group_by"] is None
    assert by_case.measurement_fingerprint() != by_slice.measurement_fingerprint()


def test_shuffle_and_validation_are_NOT_recorded_and_that_is_the_decision():
    """`pyplatypus#450` named three fields and the measurement says one. Asserted, not left.

    This is the shape the detection test uses for `ignore_threshold` one function up: a field
    deliberately out of the fingerprint is pinned so that **adding it later is a decision
    rather than a slip**, and so that the reasoning is somewhere a reader can find it.

    `shuffle` is training only - its own description says *"Only the training data: validation
    is read in order so that two epochs measure the same thing in the same way"*, and every
    `shuffle=self.spec.data.shuffle` in the four engines is on the `"train"` split while every
    scoring and predicting loader passes `shuffle=False`. So it changes what the weights
    learnt, not how they were scored.

    `validation` gates whether a validation split EXISTS rather than changing a figure's
    value: `validation: false` makes asking for one an `EngineError` that names the field, so
    there is no number to misread.
    """
    data = _data(
        train_path="a", validation_path=None, split={"fractions": [0.8, 0.2], "group_by": None}
    )
    recorded = data.measurement_fingerprint()

    assert "shuffle" not in recorded
    assert "validation" not in recorded
    # Both ARE fields, so their absence is a choice rather than a spelling mistake.
    assert {"shuffle", "validation"} <= set(type(data).model_fields)


def test_split_and_SPLITS_are_different_facts_one_letter_apart_in_one_block():
    """The two halves of `measured_with` now hold `split` and `splits`, and they are not kin.

    `splits` is the model's TILING - a source image read at `splits * input_shape` and cut,
    `pyplatypus#427`. `split` is the data's PARTITION - which cases were held out,
    `pyplatypus#450`. One decides which pictures a metric sees inside a case; the other
    decides which cases there are. A reader meeting both in one dictionary will read one for
    the other, so this pins that they are both there, that they differ, and that neither
    half of the fingerprint claims the other's key.
    """
    data = _data(
        train_path="a", validation_path=None, split={"fractions": [0.8, 0.2], "group_by": None}
    )
    model = _segmentation([{"name": "dice"}], splits=[4, 4])

    their_half = data.measurement_fingerprint()
    its_half = model.measurement_fingerprint()

    assert "split" in their_half and "split" not in its_half
    assert "splits" in its_half and "splits" not in their_half
    # And so the clash guard in `export_weights` still has nothing to fire on, which is the
    # control for a pair of keys this similar.
    assert set(their_half) & set(its_half) == set()
