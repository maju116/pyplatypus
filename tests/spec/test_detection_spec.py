"""`task` as a discriminator, and the detection half it selects.

Two things are being asserted here and they are not the same thing. One is that a
detection configuration is accepted and says what it means. The other - the reason the
union exists rather than one class with optional halves - is that a configuration mixing
the two tasks is **refused, by name, against the task that was asked for**. A single
class with every field optional would accept both of these:

    task: detection    ... colormap: [[0,0,0],[255,255,255]]
    task: segmentation ... anchors: [[[0.1, 0.1]]]

and quietly ignore the half that did not apply, which is how someone spends an afternoon
wondering why their anchors changed nothing.
"""

import pytest

from pyplatypus import ConfigError, DetectionSpec, SegmentationSpec, Task, from_dict
from pyplatypus.spec import DetectionData, DetectionModel

COCO_IN_PIXELS = [
    [[116, 90], [156, 198], [373, 326]],
    [[30, 61], [62, 45], [59, 119]],
    [[10, 13], [16, 30], [33, 23]],
]


@pytest.fixture
def detection_data():
    return {"train_path": ".", "validation_path": ".",
            "classes": ["RBC", "WBC", "Platelets"]}


@pytest.fixture
def detection_config(detection_data):
    return {"task": "detection", "data": detection_data,
            "models": [{"name": "bccd", "input_shape": [416, 416]}]}


def build(config):
    return from_dict(config, check_paths=False)


# --- the discriminator ------------------------------------------------------------------

def test_a_configuration_without_a_task_is_a_segmentation_one(config):
    """Every configuration written before detection existed is a segmentation one, and
    there are released versions of the R package that send no `task` at all."""
    spec = build(config)
    assert isinstance(spec, SegmentationSpec)
    assert spec.task is Task.SEGMENTATION
    assert spec.to_dict()["task"] == "segmentation"


def test_the_task_selects_the_detection_shape(detection_config):
    spec = build(detection_config)
    assert isinstance(spec, DetectionSpec)
    assert spec.task is Task.DETECTION
    assert isinstance(spec.data, DetectionData)
    assert isinstance(spec.models[0], DetectionModel)


def test_an_unknown_task_lists_the_two_that_exist(detection_config):
    with pytest.raises(ConfigError) as caught:
        build({**detection_config, "task": "detction"})
    message = str(caught.value)
    assert "'segmentation', 'detection'" in message
    # pydantic renders an enum tag with repr; the value is what a user types, and the
    # class name is an implementation detail that would read as the thing to write.
    assert "Task." not in message


def test_a_segmentation_field_in_a_detection_spec_is_refused(detection_config,
                                                             detection_data):
    with pytest.raises(ConfigError) as caught:
        build({**detection_config,
               "data": {**detection_data, "colormap": [[0, 0, 0], [255, 255, 255]]}})
    problem = caught.value.problems[0]
    assert problem["where"] == "data.colormap"
    assert "not permitted" in problem["problem"]


def test_a_detection_field_in_a_segmentation_spec_is_refused(config):
    models = [{**config["models"][0], "anchors": [[[0.1, 0.1]]]}]
    with pytest.raises(ConfigError) as caught:
        build({**config, "models": models})
    assert caught.value.problems[0]["where"] == "models[0].anchors"


def test_the_tag_is_not_reported_as_a_field(detection_config):
    """pydantic puts the matched tag at the head of every nested location, so an error
    reads 'detection.models[0]' - which looks like a field called `detection`."""
    with pytest.raises(ConfigError) as caught:
        build({**detection_config,
               "models": [{"name": "bccd", "input_shape": [400, 400]}]})
    assert caught.value.problems[0]["where"] == "models[0]"


def test_the_base_class_cannot_be_built_directly(config):
    from pyplatypus.spec.spec import PlatypusSpec

    with pytest.raises(ValueError, match="shared base"):
        PlatypusSpec(task="segmentation", **config)


# --- what detection needs that segmentation does not ------------------------------------

def test_classes_decide_the_count_and_the_index(detection_config):
    spec = build(detection_config)
    assert spec.n_class == 3
    assert spec.data.classes[0] == "RBC"


def test_one_class_is_allowed_because_there_is_no_background(detection_data):
    """A segmentation colormap needs at least two entries, one of them the background. A
    region of an image holding no object is not a box, so a single-class detector is an
    ordinary thing to want."""
    spec = build({"task": "detection", "data": {**detection_data, "classes": ["cell"]},
                  "models": [{"name": "one", "input_shape": [416, 416]}]})
    assert spec.n_class == 1


def test_duplicate_class_names_are_refused(detection_config, detection_data):
    with pytest.raises(ConfigError, match="distinct"):
        build({**detection_config,
               "data": {**detection_data, "classes": ["RBC", "RBC"]}})


def test_anchors_default_to_being_fitted(detection_config):
    spec = build(detection_config)
    assert spec.models[0].anchors is None
    assert spec.models[0].anchors_as_tuples is None
    assert spec.models[0].anchors_per_grid == 3


def test_anchors_are_fractions_and_pixels_say_so(detection_config):
    """The published COCO anchors are in pixels at a 416 input, so this is the mistake
    somebody makes once. The message has to name the conversion, not just the range."""
    with pytest.raises(ConfigError) as caught:
        build({**detection_config,
               "models": [{"name": "bccd", "input_shape": [416, 416],
                           "anchors": COCO_IN_PIXELS}]})
    assert "times the input size" in str(caught.value)


def test_anchors_in_fractions_are_accepted_and_come_back_as_tuples(detection_config):
    fractions = [[[w / 416, h / 416] for w, h in group] for group in COCO_IN_PIXELS]
    spec = build({**detection_config,
                  "models": [{"name": "bccd", "input_shape": [416, 416],
                              "anchors": fractions}]})
    anchors = spec.models[0].anchors_as_tuples
    assert len(anchors) == 3
    assert all(len(group) == 3 for group in anchors)
    assert anchors[0][0] == pytest.approx((116 / 416, 90 / 416))


def test_the_wrong_number_of_grids_is_refused(detection_config):
    with pytest.raises(ConfigError, match="3 groups"):
        build({**detection_config,
               "models": [{"name": "bccd", "input_shape": [416, 416],
                           "anchors": [[[0.1, 0.1]], [[0.2, 0.2]]]}]})


def test_grids_with_different_anchor_counts_are_refused(detection_config):
    """The head is one tensor of width anchors_per_grid * (n_class + 5), so this cannot
    be built at all - and the error belongs here rather than inside torch."""
    uneven = [[[0.5, 0.5], [0.4, 0.4]], [[0.2, 0.2]], [[0.1, 0.1]]]
    with pytest.raises(ConfigError, match="same number of anchors"):
        build({**detection_config,
               "models": [{"name": "bccd", "input_shape": [416, 416], "anchors": uneven}]})


def test_anchors_and_anchors_per_grid_must_agree(detection_config):
    given = [[[0.5, 0.5], [0.4, 0.4], [0.3, 0.3]]] * 3
    with pytest.raises(ConfigError, match="anchors_per_grid"):
        build({**detection_config,
               "models": [{"name": "bccd", "input_shape": [416, 416],
                           "anchors": given, "anchors_per_grid": 2}]})


@pytest.mark.parametrize("size", [[400, 400], [416, 400], [417, 416], [480, 450]])
def test_the_input_must_divide_by_the_coarsest_stride(detection_config, size):
    """The three grids are the input over 32, 16 and 8, so a side that is not a multiple
    of 32 gives a coarsest grid that does not tile the image. One side is enough to
    refuse - [416, 400] is the asymmetric case, which is the one a square check misses."""
    with pytest.raises(ConfigError, match="divisible by 32"):
        build({**detection_config, "models": [{"name": "bccd", "input_shape": size}]})


@pytest.mark.parametrize("size", [[416, 416], [608, 608], [64, 64], [320, 416]])
def test_sizes_that_do_divide_are_accepted(detection_config, size):
    spec = build({**detection_config, "models": [{"name": "bccd", "input_shape": size}]})
    assert tuple(spec.models[0].input_shape) == tuple(size)


def test_detection_in_three_dimensions_is_refused(detection_config):
    """Refused while the spec is read, like `encoder` at rank 3: nothing is allocated and
    no weights are fetched before the answer is known."""
    with pytest.raises(ConfigError, match="detection is 2D"):
        build({**detection_config,
               "models": [{"name": "bccd", "input_shape": [64, 64, 32]}]})


def test_coordinates_default_to_the_voc_convention(detection_config):
    spec = build(detection_config)
    assert spec.data.coordinates is None
    assert spec.data.voc_coordinates == "voc"


def test_coordinates_cannot_be_set_for_labelme(detection_config, detection_data):
    """LabelMe stores continuous pixel coordinates, so accepting the field and ignoring it
    would leave someone believing they had changed how their boxes are read."""
    with pytest.raises(ConfigError, match="continuous pixel coordinates"):
        build({**detection_config,
               "data": {**detection_data, "annotation_format": "labelme",
                        "coordinates": "zero_based"}})


def test_labelme_without_coordinates_is_fine(detection_config, detection_data):
    spec = build({**detection_config,
                  "data": {**detection_data, "annotation_format": "labelme"}})
    assert spec.data.annotation_format == "labelme"


def test_the_annotation_subdirectory_is_not_called_masks(detection_config):
    """A detection sample's second subdirectory holds XML or JSON, so the default differs
    from segmentation's - which is the whole reason `subdirs` has no default on the base."""
    spec = build(detection_config)
    assert spec.data.subdirs == ("images", "annotations")


# --- what both tasks share ---------------------------------------------------------------

def test_the_shared_fields_are_shared(detection_config):
    spec = build({**detection_config,
                  "seed": 7,
                  "models": [{"name": "bccd", "input_shape": [416, 416],
                              "epochs": 150, "batch_size": 4,
                              "optimizer": {"name": "adamw", "learning_rate": 1e-4}}]})
    model = spec.models[0]
    assert (spec.seed, model.epochs, model.batch_size) == (7, 150, 4)
    assert model.optimizer.name == "adamw"
    assert model.rank == 2


def test_model_names_must_be_unique_at_either_task(detection_config):
    """The validator lives on the shared base, so this is really a test that the base's
    validators run for a detection spec at all."""
    twice = [{"name": "same", "input_shape": [416, 416]}] * 2
    with pytest.raises(ConfigError, match="must be unique"):
        build({**detection_config, "models": twice})


def test_fit_false_reaches_the_dead_end_in_one_step(detection_config):
    """The base's validator would say "fit=false only makes sense together with weights",
    and adding weights to satisfy that would arrive at the refusal below. Detection
    replaces it by name so there is one message, not a trail of two."""
    with pytest.raises(ConfigError) as caught:
        build({**detection_config,
               "models": [{"name": "bccd", "input_shape": [416, 416], "fit": False}]})
    assert "fit=false: loading weights into a detector" in str(caught.value)


def test_segmentation_keeps_its_own_message(config):
    """The override is on the detection model only, so replacing the validator by name
    must not reach across."""
    models = [{**config["models"][0], "fit": False}]
    with pytest.raises(ConfigError, match="fit=false only makes sense"):
        build({**config, "models": models})


def test_a_callback_can_only_watch_the_loss(detection_config):
    """Detection has no `metrics` field, so there is nothing else to watch yet. The
    message has to list what there is rather than say no."""
    with pytest.raises(ConfigError, match="val_dice"):
        build({**detection_config,
               "models": [{"name": "bccd", "input_shape": [416, 416],
                           "callbacks": [{"name": "early_stopping",
                                          "monitor": "val_dice"}]}]})
    spec = build({**detection_config,
                  "models": [{"name": "bccd", "input_shape": [416, 416],
                              "callbacks": [{"name": "early_stopping",
                                             "monitor": "val_loss"}]}]})
    assert spec.models[0].callbacks[0].monitor == "val_loss"


def test_detection_offers_no_loss_or_metric_to_choose(detection_config):
    """YOLOv3's objective is part of the architecture and mean average precision is not
    one option among several. A field that accepts a value and ignores it is worse than
    no field, so neither exists - and that has to stay deliberate rather than pending."""
    for field, value in (("loss", {"name": "dice"}), ("metrics", [{"name": "iou"}])):
        with pytest.raises(ConfigError) as caught:
            build({**detection_config,
                   "models": [{"name": "bccd", "input_shape": [416, 416], field: value}]})
        assert caught.value.problems[0]["where"] == f"models[0].{field}"


def test_thresholds_are_separate_because_they_answer_different_questions(detection_config):
    spec = build(detection_config)
    model = spec.models[0]
    assert model.score_threshold == 0.01       # what is computed at all
    assert model.operating_point == 0.5        # where precision and recall are read
    assert model.nms_threshold == 0.45         # when two boxes are one object
    assert model.ignore_threshold == 0.5       # which cells go unsupervised


# --- the engine does not pretend ---------------------------------------------------------

def test_the_engine_refuses_a_detection_spec(detection_config, tmp_path):
    """The spec validates and nothing trains from it yet. Saying so plainly beats failing
    somewhere inside the data pipeline, and beats a feature that is advertised and does
    not work."""
    from pyplatypus import Engine
    from pyplatypus.engine import EngineError

    spec = build({**detection_config,
                  "data": {"train_path": str(tmp_path), "validation_path": str(tmp_path),
                           "classes": ["RBC"]}})
    with pytest.raises(EngineError, match="trains segmentation"):
        Engine(spec)
