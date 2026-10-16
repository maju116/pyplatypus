"""The third engine: a wider head, a composite objective, and a score that needs a decoding.

What is being asserted here, in the order the engine does it.

The **head** is `f(separation, n_class)` and comes from the specification, so a `rim` model
is three channels wide and a `flows` model four on data that names two classes. Nothing
states it, which is the whole point: the one line that forced
`examples/train_instance_flows.py` outside `Engine` was `build_model(spec, n_class=4)`.

The **objective** is composite for `flows` - a masked regression on the vectors plus an
overlap loss on the classes - and the pieces that make that work are each tested on their
own as well as through a run: the weight, the channel split, the foreground the loader
carries, and the activation that must leave a direction alone.

The **score** happens after a post-process, so `evaluate` has a row per (model, decoding),
and where a model stayed silent about `decode` the pair it derives ends in
`connected_components`. That is not symmetry for its own sake: the yardstick row is what
says whether a separation signal did anything, and on the data this was all measured on it
is the row that wins. **This paragraph said the yardstick is in every row whether it was
asked for or not**, which `#225` corrected in three other places and not in this one - a
model that writes its own `decode` list gets the list it wrote.

Beside the instance score is **the mask's own**, and which column that is was itself a
defect: `dice` reduces over the class channels a head has, so it is not one quantity across
two signals. `foreground_dice` is, and the two tests on it say what it has to be - the same
arithmetic for `rim` and `flows`, and one number per model rather than per decoding
(`#222`).
"""

import numpy as np
import pytest
import torch

from pyplatypus import build_engine, from_dict
from pyplatypus.engine import Engine, EngineError
from pyplatypus.instance_engine import (
    InstanceEngine,
    InstanceObjective,
    InstanceTrainer,
    _MetricOnClasses,
)
from pyplatypus.weights import WeightsError
from tests.conftest import write_png

SIZE = 64


@pytest.fixture
def instance_root(tmp_path):
    """Four cases, one image and three non-overlapping objects each.

    The objects are 10x10 blocks at 64x64, which survives the resize to 32 the tiling test
    needs and keeps `inspect_objects` away from the vanishing case that has its own test.
    Two of the three touch, so `connected_components` and a separation decoding cannot give
    the same answer - a fixture where every object is isolated would let the yardstick row
    score identically and prove nothing about the axis.
    """

    def build(root, cases):
        for n in range(cases):
            case = root / f"case_{n}"
            image = np.zeros((SIZE, SIZE, 3), np.uint8)
            image[5:35, 5:25] = 200
            image[40:50, 40:50] = 150
            write_png(case / "images" / f"{n}.png", image)
            for index, (rows, columns) in enumerate(
                [
                    (slice(5, 20), slice(5, 25)),
                    (slice(20, 35), slice(5, 25)),
                    (slice(40, 50), slice(40, 50)),
                ]
            ):
                mask = np.zeros((SIZE, SIZE), np.uint8)
                mask[rows, columns] = 255
                write_png(case / "masks" / f"{index}.png", mask)
        return root

    return build(tmp_path / "train", 4), build(tmp_path / "valid", 2)


@pytest.fixture
def instance_config(instance_root):
    train, valid = instance_root
    return {
        "task": "instance_segmentation",
        "seed": 1,
        "data": {"train_path": str(train), "validation_path": str(valid)},
        "models": [
            {
                "name": "nuclei",
                "input_shape": [SIZE, SIZE],
                "separation": {"name": "rim"},
                "blocks": 2,
                "filters": 4,
                "epochs": 1,
                "batch_size": 2,
            }
        ],
    }


def spec_of(config, **fields):
    models = [{**config["models"][0], **fields}]
    return from_dict({**config, "models": models})


# --- which engine, and the refusals ------------------------------------------------------


def test_build_engine_returns_the_instance_engine(instance_config):
    engine = build_engine(spec_of(instance_config))
    assert isinstance(engine, InstanceEngine)


def test_the_instance_engine_refuses_a_semantic_spec(config, data_block):
    from pyplatypus import from_dict as build

    spec = build(config, check_paths=False)
    with pytest.raises(EngineError, match="trains instance segmentation"):
        InstanceEngine(spec)


def test_the_semantic_engine_points_at_this_one(instance_config):
    """The message used to say nothing trains instances. It does now, and the message is
    the only thing that tells a reader which engine to reach for."""
    with pytest.raises(EngineError) as caught:
        Engine(spec_of(instance_config))
    assert "InstanceEngine" in str(caught.value)


# --- the dataset and the loader ----------------------------------------------------------


@pytest.mark.parametrize(
    ("separation", "channels"),
    [("rim", 3), ("flows", 4)],
)
def test_the_target_is_as_wide_as_the_head_the_spec_derives(instance_config, separation, channels):
    spec = spec_of(instance_config, separation={"name": separation})
    engine = InstanceEngine(spec)
    image, target = engine.dataset(spec.models[0], "train")[0]
    assert image.shape == (SIZE, SIZE, 3)
    assert target.shape == (SIZE, SIZE, channels)
    assert target.shape[-1] == spec.models[0].head_channels(spec.data.n_class)


def test_the_rim_target_holds_three_classes_and_the_rim_is_the_third(instance_config):
    spec = spec_of(instance_config, separation={"name": "rim", "width": 2})
    engine = InstanceEngine(spec)
    _, target = engine.dataset(spec.models[0], "train")[0]
    classes = np.argmax(target, axis=-1)
    assert set(np.unique(classes).tolist()) == {0, 1, 2}
    # The rim surrounds the interior, so it cannot be the larger of the two.
    assert int((classes == 2).sum()) < int((classes == 1).sum())


def test_the_flows_target_keeps_the_vectors_in_front_of_the_classes(instance_config):
    spec = spec_of(instance_config, separation={"name": "flows"})
    engine = InstanceEngine(spec)
    _, target = engine.dataset(spec.models[0], "train")[0]
    foreground = target[..., 3] > 0.5
    # The vectors are non-zero only on the objects, and zero at each object's sink - which
    # is exactly why the foreground is read from the class channel and not from them.
    assert not target[..., :2][~foreground].any()
    assert target[..., :2][foreground].any()
    assert int(foreground.sum()) > int((np.abs(target[..., :2]).sum(axis=-1) > 0).sum())


def test_the_loader_carries_the_foreground_for_flows_and_not_for_rim(instance_config):
    rim = spec_of(instance_config, separation={"name": "rim"})
    flows = spec_of(instance_config, separation={"name": "flows"})
    rim_batch = next(iter(InstanceEngine(rim).loader(rim.models[0], "train")))
    flows_batch = next(iter(InstanceEngine(flows).loader(flows.models[0], "train")))
    assert len(rim_batch) == 2
    assert len(flows_batch) == 3
    image, target, foreground = flows_batch
    assert foreground.shape == (image.shape[0], 1, SIZE, SIZE)
    # Channel 3 is the object class; channel 0 is a flow component. Reading background off
    # channel 0 would call almost the whole image foreground, which is the defect
    # `foreground_of` exists to prevent, so the two are compared rather than asserted loose.
    assert torch.equal(foreground[:, 0], (target[:, 3] > 0.5).float())
    assert foreground.mean() < 0.5


@pytest.fixture
def tiny_objects(tmp_path):
    """Cases whose objects are one pixel each, which is what makes them able to vanish.

    The fixture above cannot test this and that is the point of having two: its objects are
    15x20 of a 64-pixel image, so nearest-neighbour sampling still lands on all three at
    4x4 - measured. A nucleus in this dataset runs from four pixels up, and those are the
    ones a large downscale removes.
    """
    for root, cases in ((tmp_path / "small_train", 3), (tmp_path / "small_valid", 1)):
        for n in range(cases):
            case = root / f"case_{n}"
            write_png(case / "images" / f"{n}.png", np.zeros((SIZE, SIZE, 3), np.uint8))
            for index, position in enumerate([(1, 1), (3, 3), (5, 5)]):
                mask = np.zeros((SIZE, SIZE), np.uint8)
                mask[position] = 255
                write_png(case / "masks" / f"{index}.png", mask)
    return tmp_path / "small_train", tmp_path / "small_valid"


def test_a_split_whose_objects_vanish_is_refused_naming_the_input_shape(
    instance_config, tiny_objects
):
    """Specific to this task and silent without the check: the objects are on disk, the
    resize removes them, and every loss and metric stays finite while the run learns to
    answer "nothing here"."""
    train, valid = tiny_objects
    config = {
        **instance_config,
        "data": {"train_path": str(train), "validation_path": str(valid)},
    }
    # The premise, asserted rather than assumed: at the model's own size the objects are
    # there, so the refusal below is about the resize and not about an empty fixture.
    full = spec_of(config)
    assert InstanceEngine(full).dataset(full.models[0], "train").inspect_objects().objects == 9

    spec = spec_of(config, input_shape=[4, 4], blocks=1)
    with pytest.raises(EngineError) as caught:
        InstanceEngine(spec).fit()
    message = str(caught.value)
    assert "input_shape=(4, 4)" in message
    assert "splits" in message


def test_some_cases_losing_their_objects_is_a_warning_rather_than_a_refusal(
    instance_config, tiny_objects
):
    """A split that lost every object cannot train; one that lost some of them can, and
    those cases teach the model to find nothing. Different failures, different answers -
    and the warning names the count, because that is the number that says which it is."""
    train, _ = tiny_objects
    case = train / "case_big"
    write_png(case / "images" / "big.png", np.zeros((SIZE, SIZE, 3), np.uint8))
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[8:40, 8:40] = 255
    write_png(case / "masks" / "0.png", mask)

    config = {**instance_config, "data": {"train_path": str(train), "validation": False}}
    spec = spec_of(config, input_shape=[4, 4], blocks=1)
    with pytest.warns(UserWarning, match="no object left"):
        InstanceEngine(spec).fit()


# --- the composite objective, in pieces --------------------------------------------------


def test_the_objective_weighs_the_vector_term_and_splits_the_channels():
    """Built with two stubs, so the arithmetic is the only thing under test and a wrong
    channel split cannot hide behind two real losses that both fall."""
    seen = {}

    def semantic(logits, target):
        seen["semantic"] = (logits.shape[1], target.shape[1])
        return torch.tensor(2.0)

    def regression(logits, target, mask):
        seen["regression"] = (logits.shape[1], target.shape[1], mask.shape[1])
        return torch.tensor(5.0)

    objective = InstanceObjective(semantic, regression, weight=3.0, semantic_from=2)
    logits = torch.zeros(1, 4, 8, 8)
    target = torch.zeros(1, 4, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    assert float(objective(logits, target, mask)) == pytest.approx(3.0 * 5.0 + 2.0)
    assert seen["regression"] == (2, 2, 1)
    assert seen["semantic"] == (2, 2)
    assert objective.needs_mask is True


def test_a_metric_on_a_flows_model_sees_only_the_class_channels():
    """A metric handed the whole head would softmax across two directions and two classes.
    The wrapper is checked against a metric that reports what it was shown, because a real
    metric's number moving is not evidence about which channels it read."""

    class Reports(torch.nn.Module):
        accumulates = True

        def forward(self, logits, target):
            return torch.tensor(float(logits.shape[1]))

    wrapped = _MetricOnClasses(Reports(), start=2)
    assert float(wrapped(torch.zeros(1, 4, 4, 4), torch.zeros(1, 4, 4, 4))) == 2.0
    assert wrapped.accumulates is True

    class WholeMask(Reports):
        accumulates = False

    # Proxied, not answered for: the trainer reads this to keep a whole-mask metric out of
    # a tiled epoch, and a wrapper with its own answer would put it back.
    assert _MetricOnClasses(WholeMask(), start=2).accumulates is False


def test_the_activation_leaves_a_direction_alone_and_normalises_the_classes(instance_config):
    spec = spec_of(instance_config, separation={"name": "flows"})
    engine = InstanceEngine(spec)
    engine.fit()
    trainer = engine.runs["nuclei"].trainer
    out = torch.randn(2, 4, 8, 8)
    activated = trainer.activate(out)
    assert torch.equal(activated[:, :2], out[:, :2])
    assert torch.allclose(activated[:, 2:].sum(dim=1), torch.ones(2, 8, 8), atol=1e-5)


def test_a_rim_model_activates_the_whole_head(instance_config):
    spec = spec_of(instance_config, separation={"name": "rim"})
    # A real module, because a trainer builds an optimizer over its parameters and
    # `nn.Identity` has none - the activation is what is under test, not the network.
    trainer = InstanceTrainer(torch.nn.Conv2d(3, 3, 1), spec.models[0], device="cpu")
    out = torch.randn(1, 3, 4, 4)
    assert torch.allclose(trainer.activate(out).sum(dim=1), torch.ones(1, 4, 4), atol=1e-5)


# --- fit, and the head it builds ---------------------------------------------------------


@pytest.mark.parametrize("separation", ["rim", "flows"])
def test_fit_trains_a_head_as_wide_as_the_separation_asks(instance_config, separation):
    spec = spec_of(instance_config, separation={"name": separation})
    engine = InstanceEngine(spec)
    histories = engine.fit()
    assert len(histories["nuclei"]) == 1
    run = engine.runs["nuclei"]
    assert run.trained
    with torch.no_grad():
        out = run.model(torch.zeros(1, 3, SIZE, SIZE).to(run.trainer.device))
    assert out.shape[1] == spec.models[0].head_channels(spec.data.n_class)
    assert "train_loss" in run.history.records[0]


def test_a_flows_run_reports_the_mask_rather_than_the_vectors(instance_config):
    """The metrics are on the class channels, so `train_iou` is the mask's own number -
    which is the first thing to read: if the mask moved, the instance score moved for a
    reason that is not separation."""
    engine = InstanceEngine(spec_of(instance_config, separation={"name": "flows"}))
    engine.fit()
    record = engine.runs["nuclei"].history.records[0]
    assert "train_iou" in record
    assert 0.0 <= record["train_iou"] <= 1.0


# --- evaluate: a row per (model, decoding) ----------------------------------------------


def test_a_flows_models_metric_is_perfect_on_a_perfect_mask(instance_config):
    """The behavioural half of `_MetricOnClasses`, and the reason the wrapper exists.

    Handed the target as the prediction, a metric over the class channels reports 1.0. The
    same metric over the whole head cannot: the two flow channels are not a one-hot of
    anything, so a softmax across all four moves probability out of the class channels and
    the number comes back below 1 - plausible, bounded, and not a measurement of the mask.
    """
    spec = spec_of(instance_config, separation={"name": "flows"})
    engine = InstanceEngine(spec)
    engine.fit()
    trainer = engine.runs["nuclei"].trainer
    _, target = engine.dataset(spec.models[0], "train")[0]
    perfect = torch.from_numpy(np.moveaxis(target, -1, 0))[None]
    # A one-hot target is not logits, so the class block is scaled up until its softmax is
    # effectively the one-hot it stands for; the point is which channels are read.
    logits = perfect * 20.0

    wrapped = float(trainer.metrics["iou"](logits, perfect))
    whole_head = float(trainer.metrics["iou"].metric(logits, perfect))
    assert wrapped == pytest.approx(1.0, abs=1e-3)
    assert whole_head < wrapped


def test_pooling_by_group_renumbers_before_it_stacks(instance_config):
    """Two cases both number their objects from 1. Stacked without renumbering, object 1 of
    each becomes one object spanning both - so a group of two three-object cases would
    report three objects instead of six, and the score would be computed against a truth
    that merged them."""
    engine = InstanceEngine(spec_of(instance_config))
    engine.fit()
    rows = engine.evaluate_cases(group_by=r"case")
    assert len(rows) == 1
    assert rows[0]["group"] == "case"
    assert rows[0]["truth"] == 6


def test_evaluate_gives_a_row_per_model_and_decoding(instance_config):
    engine = InstanceEngine(spec_of(instance_config, separation={"name": "rim"}))
    engine.fit()
    table = engine.evaluate()
    assert [row["decode"] for row in table] == ["nearest_seed", "connected_components"]
    for row in table:
        assert row["model"] == "nuclei"
        assert row["separation"] == "rim"
        assert 0.0 <= row["average_precision"] <= 1.0
        # The counts beside every score, because an average precision cannot tell a route
        # that mislabels from one that shatters.
        assert row["truth_objects"] == 6  # two validation cases, three objects each
        assert "predicted_objects" in row
        # The mask's own numbers, the same on both rows of one model.
        assert "iou" in row and "loss" in row
    assert table[0]["iou"] == table[1]["iou"]


def test_the_yardstick_is_in_the_table_without_being_asked_for(instance_config):
    """`decode` was left unset, so the pair is derived - and the second of them is the row
    the first has to be read against."""
    spec = spec_of(instance_config, separation={"name": "flows"})
    assert spec.models[0].decode is None
    engine = InstanceEngine(spec)
    engine.fit()
    assert [row["decode"] for row in engine.evaluate()] == [
        "follow_flows",
        "connected_components",
    ]


def test_a_written_decode_list_is_what_you_get(instance_config):
    """The other branch, and the one nothing covered. `evaluate`'s own docstring claimed the
    yardstick was in every model's rows *whether it was asked for or not* - which is true of
    the derived pair above and false here, and the gap went unnoticed because only the kind
    half had a test.

    Asserted as the exact list rather than as an absence, so it also fails if a third row
    appears from somewhere else. `examples/compare_instance_separations.py` keeps the
    yardstick in its own `--decode` handling, which is the right place for it: an example
    whose whole claim is a comparison needs the row, and a user's table is theirs.
    """
    spec = spec_of(instance_config, separation={"name": "flows"}, decode=[{"name": "follow_flows"}])
    assert [block.name for block in spec.models[0].decode] == ["follow_flows"]
    engine = InstanceEngine(spec)
    engine.fit()
    assert [row["decode"] for row in engine.evaluate()] == ["follow_flows"]


#: The two channel layouts, written out rather than read from `Separation.semantic_from`.
#: That is the point of the test below: the engine derives this offset and the column must
#: not depend on how many channels follow it, so the test states the contract independently
#: instead of asking the code that implements it. `test_the_target_is_as_wide_as_the_head`
#: above is what holds these two numbers to the heads.
CLASS_BLOCK_STARTS = {"rim": 0, "flows": 2}


def _binary_dice(predictions, truths, start):
    """The definition, in the test's own arithmetic: foreground against foreground.

    Unsmoothed and averaged over images, which are two of the three decisions
    `foreground_dice` documents; the third is that both sides are binarised, which is the
    `> 0` on each.
    """
    scores = []
    for prediction, truth in zip(predictions, truths, strict=True):
        called = np.argmax(prediction[..., start:], axis=-1) > 0
        actual = truth > 0
        scores.append(2 * np.logical_and(called, actual).sum() / (called.sum() + actual.sum()))
    return float(np.mean(scores))


def test_the_mask_column_means_the_same_thing_for_both_separations(instance_config):
    """`foreground_dice` is comparable across signals where `dice` is not - `pyplatypus#222`.

    The defect this guards shipped in `#215`'s published table: the column read there was
    the training `dice`, a mean over the class channels a head has - three for `rim`
    including a thin rim band that scores far lower than the rest, two for `flows`. It
    showed the two masks 0.09 apart when they are 0.0062 apart, **and with the sign
    reversed**.

    So the assertion is not that the two numbers are close on this fixture, which would be a
    claim about the fixture. It is that each model's column equals a binary foreground Dice
    computed by **one piece of arithmetic applied to both**, with the channel offset written
    into the test rather than read from the engine. An implementation that reduced over the
    class channels fails here on the rim model while still passing on `flows`, which is
    exactly how the defect looked.
    """
    first = instance_config["models"][0]
    spec = from_dict(
        {
            **instance_config,
            "models": [
                {**first, "name": "rim", "separation": {"name": "rim"}, "epochs": 2},
                {**first, "name": "flows", "separation": {"name": "flows"}, "epochs": 2},
            ],
        }
    )
    assert [model.head_channels(spec.data.n_class) for model in spec.models] == [3, 4]
    engine = InstanceEngine(spec)
    engine.fit()
    table = engine.evaluate()

    for name, start in CLASS_BLOCK_STARTS.items():
        rows = [row for row in table if row["model"] == name]
        assert rows, f"no rows for {name}"
        dataset = engine.dataset(engine.runs[name].spec, "validation")
        truths = [dataset.labels_of(index) for index in range(len(dataset.samples))]
        expected = _binary_dice(engine.predict(name, split="validation"), truths, start)
        for row in rows:
            assert row["foreground_dice"] == pytest.approx(expected, abs=1e-9), (
                f"{name}'s column is {row['foreground_dice']} where the binary foreground "
                f"Dice of the same prediction is {expected}; the two signals' columns are "
                "only comparable while both are this quantity"
            )


def test_the_mask_column_is_one_number_per_model_and_not_per_decoding(instance_config):
    """The claim `evaluate`'s table makes about the column, held by the structure.

    A decoding turns channels into objects and cannot move the mask, so the number has to be
    identical on every row of one model - that is what makes it the thing to read first when
    an instance score moves. `_Decoded` carries it beside the decodings rather than inside
    each of them, and this is the test that notices if it is ever computed per decoding.
    """
    engine = InstanceEngine(
        spec_of(
            instance_config,
            separation={"name": "rim"},
            epochs=2,
            # Asked for by name, because the comparison below is the finding: this model's
            # `dice` is a mean over three class channels and the new column is not.
            metrics=[{"name": "dice"}],
        )
    )
    engine.fit()
    table = engine.evaluate()
    assert len(table) == 2, "the derived pair is what makes this test about two rows"
    assert table[0]["foreground_dice"] is not None
    assert table[0]["foreground_dice"] == table[1]["foreground_dice"]
    # And it is not the training metric wearing a second name, which is the whole finding.
    assert table[0]["foreground_dice"] != table[0]["dice"]


def test_evaluate_refuses_before_anything_is_trained(instance_config):
    with pytest.raises(EngineError, match="nothing has been trained"):
        InstanceEngine(spec_of(instance_config)).evaluate()


def test_evaluate_cases_is_one_row_per_case_for_one_decoding(instance_config):
    engine = InstanceEngine(spec_of(instance_config))
    engine.fit()
    rows = engine.evaluate_cases()
    assert [row["case"] for row in rows] == ["case_0", "case_1"]
    assert {row["decode"] for row in rows} == {"nearest_seed"}
    for row in rows:
        assert row["truth"] == 3
        assert "predicted" in row

    other = engine.evaluate_cases(decode="connected_components")
    assert {row["decode"] for row in other} == {"connected_components"}


def test_evaluate_cases_refuses_a_decoding_the_model_does_not_ask_for(instance_config):
    engine = InstanceEngine(spec_of(instance_config))
    engine.fit()
    with pytest.raises(EngineError) as caught:
        engine.evaluate_cases(decode="watershed")
    message = str(caught.value)
    assert "watershed" in message
    assert "nearest_seed" in message


def test_best_model_searches_both_axes(instance_config):
    engine = InstanceEngine(spec_of(instance_config))
    engine.fit()
    assert engine.best_model() == "nuclei"
    with pytest.raises(EngineError, match="no column"):
        engine.best_model(key="f1")


# --- predict -----------------------------------------------------------------------------


def test_predict_leaves_the_vectors_raw(instance_config):
    engine = InstanceEngine(spec_of(instance_config, separation={"name": "flows"}))
    engine.fit()
    predictions = engine.predict(split="validation")
    assert predictions.shape == (2, SIZE, SIZE, 4)
    classes = predictions[..., 2:]
    assert np.allclose(classes.sum(axis=-1), 1.0, atol=1e-5)
    # A softmax over the whole head would have made all four sum to one, which is the
    # defect this is here to catch rather than a property worth having.
    assert not np.allclose(predictions.sum(axis=-1), 1.0, atol=1e-3)


def test_predict_instances_gives_one_label_map_per_case(instance_config):
    engine = InstanceEngine(spec_of(instance_config))
    engine.fit()
    maps = engine.predict_instances(split="validation")
    assert len(maps) == 2
    for labels in maps:
        assert labels.shape == (SIZE, SIZE)
        assert labels.min() >= 0
    assert engine.predict_instances(split="validation", decode="connected_components")


# --- weights -----------------------------------------------------------------------------


def test_the_sidecar_records_the_separation_and_the_task(instance_config, tmp_path):
    engine = InstanceEngine(spec_of(instance_config, separation={"name": "flows"}))
    engine.fit()
    path = engine.export_weights("nuclei", tmp_path / "flows.safetensors")
    import json

    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["separation"] == "flows"
    assert sidecar["task"] == "instance_segmentation"


def test_weights_from_one_separation_are_refused_by_the_other(instance_config, tmp_path):
    """Two heads of the same architecture that mean entirely different things. At two
    classes the widths differ as well, so the shapes would catch it - the point of the
    fingerprint is that it says *why* before torch says *what*."""
    flows = InstanceEngine(spec_of(instance_config, separation={"name": "flows"}))
    flows.fit()
    path = flows.export_weights("nuclei", tmp_path / "flows.safetensors")

    rim = spec_of(instance_config, separation={"name": "rim"}, weights=str(path))
    with pytest.raises(WeightsError) as caught:
        InstanceEngine(rim).fit()
    assert "separation" in str(caught.value)


# --- what the file may fill in -------------------------------------------------------
#
# `#236`. The semantic engine has adopted a published file's geometry since `0.8.0a13`,
# and this one explained the absence of the same mechanism by *there are no published
# instance weights* - a sentence `dsbowl-instances` ended one release later. The four
# tests below are the four decisions in it: what is adopted, what is not, that a stated
# value still wins, and that the permission to stay silent is paid for by a refusal when
# the file turns out to say nothing.


def model_without(config, *dropped, **fields):
    """The fixture's model with some keys **absent** rather than `None`.

    Not `spec_of(..., separation=None)`: pydantic counts a key that was passed at all as
    set, so an explicit null would be a stated value and adoption would correctly leave it
    alone. What a reader writes is a configuration with no such line, and that is the thing
    worth testing - and the thing R sends, since `compact()` drops an unset option.
    """
    model = {
        key: value for key, value in {**config["models"][0], **fields}.items() if key not in dropped
    }
    return from_dict({**config, "models": [model]})


@pytest.fixture
def published(instance_config, tmp_path):
    """A rim model at width 2, exported the way `export_weights` writes one.

    Width 2 rather than the default, so that an adopted block can be told from a block
    that was there anyway - the failure this mechanism invites is adopting the *name* and
    defaulting the setting, which reads identical on a fixture that uses the default.
    """
    engine = InstanceEngine(spec_of(instance_config, separation={"name": "rim", "width": 2}))
    engine.fit()
    return engine.export_weights("nuclei", tmp_path / "rim.safetensors")


def test_the_sidecar_records_the_settings_and_not_only_the_signal(published):
    """The adoption's source, written by the engine rather than by whoever publishes.

    `dsbowl-instances` carries `separation_settings` because its publishing script added
    it by hand; nothing else that ever exported an instance model did, so an adoption
    reading that key would have worked for exactly one file in the world.
    """
    import json

    sidecar = json.loads(published.with_suffix(".json").read_text())
    assert sidecar["separation"] == "rim", "the signal is part of what the weights are for"
    assert sidecar["separation_settings"]["name"] == "rim"
    assert sidecar["separation_settings"]["width"] == 2


def test_a_file_fills_in_the_four_fields_a_reader_cannot_guess(instance_config, published, capsys):
    """The issue's own example: a name, a shape, the weights, and the decoding you want.

    `separation` is the one that matters here. The other three are a published model's
    geometry and the semantic engine has adopted them for releases; a rim's width is a
    training decision, so this is the first adopted value that is not a shape - which is
    the argument for printing what was taken.
    """
    spec = model_without(
        instance_config,
        "separation",
        "blocks",
        "filters",
        weights=str(published),
        fit=False,
    )
    model = spec.models[0]
    assert model.separation is None and "blocks" not in model.model_fields_set

    engine = InstanceEngine(spec)
    engine.fit(verbose=True)
    adopted = engine.runs["nuclei"].spec

    assert adopted.separation is not None
    assert adopted.signal.value == "rim"
    assert adopted.separation.width == 2, "the settings, not only the name"
    assert (adopted.blocks, adopted.filters) == (2, 4)
    printed = capsys.readouterr().out
    assert "adopted from" in printed and "separation=" in printed


def test_a_specification_cannot_be_read_before_the_weights_have_been(instance_config, published):
    """What the window between the two costs, and what it says instead of `None`.

    The spec layer permits the omission and cannot check it - resolving a registry name
    there would mean a specification that needs a download to be validated - so a model
    read but never passed through `fit` has no answer to what its head predicts. That is
    an error with a sentence in it rather than an attribute error on a field nobody wrote.
    """
    spec = model_without(instance_config, "separation", weights=str(published))
    with pytest.raises(ValueError, match="left `separation` to its weights"):
        assert spec.models[0].signal
    with pytest.raises(ValueError, match="adopts it from the sidecar"):
        spec.models[0].head_channels(2)


def test_the_separation_cannot_be_omitted_with_nothing_to_adopt_it_from(instance_config):
    """The half of the permission the specification layer can enforce by itself."""
    from pyplatypus.errors import ConfigError

    # A reader meets this through `from_dict`, which collects pydantic's complaints into
    # one message - so the refusal is asserted where it is actually read rather than on
    # the validator that writes it.
    with pytest.raises(ConfigError, match="required unless `weights`"):
        model_without(instance_config, "separation")


def test_weights_that_do_not_say_what_they_predict_are_refused_rather_than_guessed(
    instance_config, published
):
    """The other half, paid where the file can actually be read.

    A sidecar written before `0.8.0a28` carries no `separation_settings`, and neither does
    one from somebody else's project. Guessing `rim` here would load a flows head and
    predict confidently - the failure the fingerprint exists for - and defaulting to the
    fixture's own answer would make this test pass on every file.
    """
    import json

    sidecar = published.with_suffix(".json")
    older = json.loads(sidecar.read_text())
    del older["separation_settings"]
    sidecar.write_text(json.dumps(older))

    spec = model_without(instance_config, "separation", weights=str(published))
    with pytest.raises(EngineError, match="does not record it"):
        InstanceEngine(spec).fit()


def test_a_stated_separation_is_kept_even_where_the_file_disagrees(instance_config, published):
    """Adoption loosens what may be omitted and nothing about what is checked.

    A rim at width 3 against weights trained at width 2 is the ordinary way to start one
    run from another: the settings are deliberately outside the fingerprint, because they
    change what the channels were trained *towards* rather than what they are. So this
    loads, and the stated width survives - adopting over it would silently retrain
    somebody else's decision.
    """
    spec = spec_of(
        instance_config, separation={"name": "rim", "width": 3}, weights=str(published), fit=False
    )
    engine = InstanceEngine(spec)
    engine.fit()
    assert engine.runs["nuclei"].spec.separation.width == 3


def test_the_decoding_is_never_adopted(instance_config, published):
    """`decode` is the reader's question, not the file's property.

    One trained file answers it three ways - 0.5445, 0.5390 and 0.4822 on the published
    instance weights - and the sidecar lists every decoding it was measured with. Adopting
    from that list would silently pick whichever name came first and call it the model's.
    """
    import json

    sidecar = published.with_suffix(".json")
    described = json.loads(sidecar.read_text())
    described["decodings"] = ["watershed"]
    described["decode"] = ["watershed"]
    sidecar.write_text(json.dumps(described))

    spec = model_without(instance_config, "separation", "decode", weights=str(published), fit=False)
    engine = InstanceEngine(spec)
    engine.fit()
    adopted = engine.runs["nuclei"].spec
    assert adopted.decode is None, "nothing was written, so nothing is in the field"
    assert [block.name for block in adopted.decodings] == ["nearest_seed", "connected_components"]
