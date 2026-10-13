"""Reading instances to train on them, and the ordering that makes the target honest.

The dataset's whole content is an order: read, resize, tile, augment, **then** build. Three
of those four steps move the objects, and a target that encodes a direction or a distance
has to be built from where the objects ended up - `pyplatypus#210` measured what correcting
instead of rebuilding costs, and `tests/data/test_vector_targets.py` is where that
measurement lives. What is tested here is that this dataset is on the right side of it.

The other half is the pair of contracts between `targets.py` and `decode.py`: the channel
order, and the fact that a perfect prediction decodes back into the objects it was built
from. That second one is the strongest check available without training anything, and it is
the one that would catch a channel swapped between the two modules - which no shape
assertion can, because both arrays have the same shape.
"""

import numpy as np
import pytest

from pyplatypus.data.instances import InstanceDataError, InstanceDataset
from pyplatypus.instances import InstanceError, instance_target, resize_labels
from pyplatypus.instances.decode import decode_instances, read_prediction
from pyplatypus.spec import ConnectedComponents, NearestSeed, Separation
from pyplatypus.spec.instances import FollowFlows, InstanceData, InstanceModel
from tests.conftest import write_png

SIZE = 64


def three_objects(size=SIZE):
    """Two objects that touch along a seam, and one on its own.

    The touching pair is what makes this fixture able to tell the decodings apart:
    connected components cannot separate them by construction, so a test over isolated
    objects would score every route identically and prove nothing.
    """
    labels = np.zeros((size, size), np.int32)
    labels[5:20, 5:25] = 1
    labels[20:35, 5:25] = 2
    labels[45:55, 45:55] = 3
    return labels


def write_case(root, labels, name="case_0"):
    case = root / name
    write_png(case / "images" / "image.png", np.zeros((*labels.shape, 3), np.uint8))
    for value in np.unique(labels[labels > 0]):
        write_png(case / "masks" / f"{value}.png", ((labels == value) * 255).astype(np.uint8))
    return root


def model(separation=None, **fields):
    """One instance model, given a `separation` block - `{"name": "rim"}` when silent."""
    return InstanceModel(
        **{
            "name": "m",
            "input_shape": [SIZE, SIZE],
            "separation": separation or {"name": "rim"},
            "blocks": 2,
            **fields,
        }
    )


def data_for(root):
    return InstanceData(train_path=str(root), validation=False)


def dataset_for(root, spec, **kwargs):
    from pyplatypus.data.paths import discover

    return InstanceDataset(
        discover(str(root), data_for(root), strict=True).samples,
        spec,
        data_for(root),
        **kwargs,
    )


# --- identity survives the read ---------------------------------------------------------


def test_the_objects_are_read_as_themselves_rather_than_united(tmp_path):
    """`unite_masks` is what the semantic dataset does with these same files, and it is
    where identity is lost. Three files in, three objects out."""
    root = write_case(tmp_path / "train", three_objects())
    labels = dataset_for(root, model()).labels_of(0)
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2, 3]


def test_a_case_with_no_mask_files_is_refused_by_name(tmp_path):
    """Reached by handing the dataset a sample rather than by discovering one.

    Discovery gets there first on the ordinary path - a case directory with an empty
    `masks/` is "incomplete" and the split refuses before any dataset exists - so this
    guard only fires for samples somebody built, which an example script assembling the
    pieces by hand does. Written against that path rather than deleted, and the dataset
    says the same thing in a comment, because the alternative is a `list index out of
    range` from `masks[0]`.
    """
    from pyplatypus.data.paths import Sample

    root = tmp_path / "train"
    case = root / "case_0"
    write_png(case / "images" / "image.png", np.zeros((SIZE, SIZE, 3), np.uint8))
    sample = Sample(key="case_0", images=(case / "images" / "image.png",), masks=())
    dataset = InstanceDataset((sample,), model(), data_for(root))
    with pytest.raises(InstanceDataError, match="no mask files"):
        dataset.labels_of(0)

    # And the ordinary path, which is a different refusal from a different layer - worth
    # pinning so that a change to either is not read as the other having stopped working.
    (case / "masks").mkdir(parents=True)
    with pytest.raises(Exception, match="incomplete"):
        dataset_for(root, model()).labels_of(0)


def test_two_objects_claiming_one_pixel_are_refused_rather_than_resolved(tmp_path):
    """Measured on this dataset: 0 of 670 cases overlap anywhere, so an overlap is either a
    different dataset or the open representation question (`pyplatypus#158`). Either way
    picking one of the two silently is the answer not to give."""
    labels = three_objects()
    root = write_case(tmp_path / "train", labels)
    overlapping = np.zeros((SIZE, SIZE), np.uint8)
    overlapping[10:15, 10:15] = 255  # inside object 1
    write_png(root / "case_0" / "masks" / "9.png", overlapping)
    with pytest.raises(InstanceError, match="overlaps"):
        dataset_for(root, model()).labels_of(0)


def test_the_label_map_is_built_before_the_resize_not_after(tmp_path):
    """The order that keeps the refusal above from firing on its own data.

    Resizing each object's mask and labelling the stack afterwards would let two
    nearest-neighbour resizes disagree on a boundary pixel, so a dataset with no overlap on
    disk would start raising. One integer map through one resize cannot overlap at all,
    which this asserts by resizing to a size where the seam between the touching pair has
    to land somewhere.
    """
    root = write_case(tmp_path / "train", three_objects())
    small = dataset_for(root, model(input_shape=[16, 16])).labels_of(0)
    assert small.shape == (16, 16)
    assert sorted(np.unique(small).tolist()) == [0, 1, 2, 3]


def test_an_object_that_vanishes_does_not_leave_a_hole_in_the_numbering():
    labels = np.zeros((32, 32), np.int32)
    labels[0:16, 0:16] = 1
    labels[20, 20] = 2
    labels[24:28, 24:28] = 3
    small = resize_labels(labels, (8, 8))
    present = sorted(np.unique(small[small > 0]).tolist())
    # Whatever survived is numbered 1..n with no gaps, because the metric counts objects by
    # their values and a gap reads as an object the prediction missed.
    assert present == list(range(1, len(present) + 1))
    assert len(present) < 3


# --- the ordering -----------------------------------------------------------------------


class RecordingFlip:
    """A horizontal flip that records what it was handed, which is the only way to see it.

    The dtype is the evidence: an integer array means the label map reached the augmenter,
    which is the thing `pyplatypus#210` settled - a float target arriving here would be a
    flow field being carried through as a mask, and `_refuse_continuous_mask` exists
    because that failure is otherwise silent.
    """

    def __init__(self):
        self.saw = []

    def __call__(self, image, mask=None):
        self.saw.append(None if mask is None else mask.dtype)
        if mask is None:
            return image[:, ::-1], None
        return image[:, ::-1], mask[:, ::-1]


@pytest.mark.parametrize("separation", ["rim", "flows"])
def test_the_target_is_built_after_the_augmentation(tmp_path, separation):
    """Built from the flipped objects, not flipped after being built.

    For `rim` the two happen to agree - a rim is a mask and flips like one - and for
    `flows` they do not, which is why both are checked here rather than only the one that
    can go wrong: the assertion is about the route, and a route that is right for one
    target and wrong for the other is the shape of defect that survives.
    """
    labels = three_objects()
    root = write_case(tmp_path / "train", labels)
    spec = model({"name": separation})
    augmenter = RecordingFlip()
    _, target = dataset_for(root, spec, augmenter=augmenter)[0]

    assert augmenter.saw == [np.dtype(np.int32)], "the label map has to arrive as integers"
    expected = instance_target(labels[:, ::-1], spec.signal, n_class=2, **spec.target_settings)
    assert np.allclose(target, expected)

    if separation == "flows":
        # And the thing that would have gone unnoticed: flipping the built target instead
        # leaves the column component pointing the wrong way, in the same shape and range.
        wrong = instance_target(labels, spec.signal, n_class=2, **spec.target_settings)[:, ::-1]
        assert not np.allclose(target, wrong)


def test_a_tiled_dataset_tiles_the_target_and_keeps_the_whole_map_for_scoring(tmp_path):
    """Tiling cuts the target; it does not cut what the score is computed against. An
    object crossing a seam becomes two objects if the post-process runs per tile, so
    decoding happens after stitching and `labels_of` stays on the source grid."""
    root = write_case(tmp_path / "train", three_objects())
    spec = model(input_shape=[32, 32], splits=(2, 2))
    dataset = dataset_for(root, spec)
    assert len(dataset) == 4
    _, target = dataset[0]
    assert target.shape == (32, 32, 3)
    assert dataset.labels_of(0).shape == (64, 64)


# --- the channel contract, both ways ----------------------------------------------------


def test_the_foreground_comes_from_the_class_channel_rather_than_channel_zero(tmp_path):
    root = write_case(tmp_path / "train", three_objects())
    spec = model({"name": "flows"})
    dataset = dataset_for(root, spec)
    _, target = dataset[0]
    foreground = dataset.foreground_of(target)
    assert foreground.shape == (SIZE, SIZE, 1)
    assert np.array_equal(foreground[..., 0] > 0.5, target[..., 3] > 0.5)
    # Channel 0 is a flow component, and a mask read from it would be almost everything.
    assert foreground.mean() < 0.5
    # The seeds are in the foreground although the field is zero there, which is the whole
    # reason this is not read off the vectors.
    zero_vectors = np.abs(target[..., :2]).sum(axis=-1) == 0
    assert (zero_vectors & (foreground[..., 0] > 0.5)).any()


def test_a_setting_from_the_other_mechanism_is_refused_by_the_builder_too():
    """The specification refuses the combination before anything is read; the builder
    refuses it again, because an example script assembling the pieces by hand reaches the
    builder directly and would otherwise get a silently different target."""
    labels = three_objects()
    with pytest.raises(InstanceError, match="mode"):
        instance_target(labels, Separation.RIM, mode="diffusion")
    with pytest.raises(InstanceError, match="width"):
        instance_target(labels, Separation.FLOWS, width=3)


@pytest.mark.parametrize(
    ("separation", "block", "settings"),
    [
        ("rim", NearestSeed(), {"width": 2}),
        ("flows", FollowFlows(), {"mode": "diffusion"}),
    ],
)
def test_a_perfect_prediction_decodes_back_into_the_objects_it_was_built_from(
    separation, block, settings
):
    """The ceiling check, and the only one that can catch a channel swapped between the two
    modules: both arrays would still have the right shape and range.

    The target is handed to the decoding *as* the prediction, which is what a perfect model
    would produce - so this is the same measurement as the ceiling runs on `#159` and
    `#212`, at three objects instead of six thousand.
    """
    labels = three_objects()
    target = instance_target(labels, Separation(separation), n_class=2, **settings)
    decoded = decode_instances(target, Separation(separation), block)
    assert int(decoded.max()) == 3

    # And the yardstick cannot do it, which is what makes the three above a result rather
    # than an arithmetic identity: the touching pair is one region.
    yardstick = decode_instances(target, Separation(separation), ConnectedComponents())
    assert int(yardstick.max()) == 2


def test_reading_a_prediction_gives_a_decoding_only_what_it_reads():
    labels = three_objects()
    rim = read_prediction(instance_target(labels, Separation.RIM, width=2), Separation.RIM)
    assert rim.interior is not None and rim.surface is not None and rim.flow is None
    flows = read_prediction(
        instance_target(labels, Separation.FLOWS, mode="centroid"), Separation.FLOWS
    )
    assert flows.flow is not None and flows.interior is None
    assert flows.flow.shape == (2, SIZE, SIZE)


def test_a_decoding_reaching_the_wrong_signal_is_refused_there_too():
    labels = three_objects()
    target = instance_target(labels, Separation.RIM, width=2)
    with pytest.raises(InstanceError, match="follow_flows"):
        decode_instances(target, Separation.RIM, FollowFlows())


def test_the_dataset_refuses_to_be_scored_when_it_read_no_objects(tmp_path):
    root = write_case(tmp_path / "train", three_objects())
    dataset = dataset_for(root, model(), only_images=True)
    _, target = dataset[0]
    assert target is None
    with pytest.raises(InstanceDataError, match="only_images"):
        dataset.labels_of(0)


def test_the_resize_cannot_interpolate_two_objects_into_a_third():
    """Nearest-neighbour is load-bearing and the mutation that removes it had to be hunted.

    An interpolated label map averages identities: between an object numbered 1 and one
    numbered 5, a blend lands on 2, 3 or 4 - values nobody wrote - and after the
    consecutive relabel those read as **extra objects**, a false positive the resize
    invented. The gap in the numbering is what makes it visible, and a boundary that does
    not land on exact pixel centres is what makes it happen: at an even scale factor with
    adjacent numbers the two filters agree by coincidence, which is how the first version
    of this test passed with interpolation turned on.
    """
    labels = np.zeros((32, 32), np.int32)
    labels[4:16, 4:28] = 1
    labels[16:28, 4:28] = 5
    assert int(resize_labels(labels, (7, 7)).max()) == 2
    assert int(resize_labels(labels, (9, 9)).max()) == 2
    assert int(resize_labels(labels, (13, 13)).max()) == 2


def test_minimum_area_reaches_the_decoding():
    """A parameter a block carries and the decoding never receives is a setting that does
    nothing, which is the failure mode every block in this format is validated against."""
    labels = three_objects()
    target = instance_target(labels, Separation.RIM, width=2)
    assert int(decode_instances(target, Separation.RIM, NearestSeed()).max()) == 3
    # The smallest of the three objects is 10x10; a floor above that drops it and nothing
    # else, so the count says the number arrived rather than that something was dropped.
    kept = decode_instances(target, Separation.RIM, NearestSeed(minimum_area=150))
    assert int(kept.max()) == 2
