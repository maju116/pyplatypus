"""Flows: the target, the two ways of building it, and following them back to objects.

`#159` closed because an interior and a rim compete for the same pixels. A flow field does
not: every foreground pixel carries a unit vector towards its own object and nothing is
subtracted from the object to make room for the signal. These tests are about the two places
that premise can still go wrong - a field built by the cheap route, and a field followed for
too few steps - because both produce labels that look exactly like labels.

Measured on 134 held-out Data Science Bowl cases and 6000 objects before any of this was
trusted, which is the same order `parts_from_instances` was written in: the ground truth
through `diffusion` + `instances_by_flows` scores an instance AP of **0.9924** against
**0.9478** for `centroid`, where interior-plus-rim reaches 0.9807 and connected components
0.7583 on those same pictures. So the difference between the two modes is **15x** the 0.0030
this dataset could not rank four architectures inside, and the cheap mode is the one that
loses it.
"""

import numpy as np
import pytest

from pyplatypus.instances import (
    average_precision,
    flows_from_instances,
    follow_flows,
    instances_by_flows,
    instances_from_mask,
    iou_matrix,
)
from pyplatypus.instances.flows import _sink
from pyplatypus.instances.metrics import InstanceError
from pyplatypus.instances.postprocess import parts_from_instances


def touching_pair(size=(9, 11)):
    """Two blocks sharing a seam - the case connected components cannot read."""
    truth = np.zeros(size, dtype=np.int32)
    truth[2:7, 1:6] = 1
    truth[2:7, 6:11] = 2
    return truth


def crescent(size=21):
    """A C, whose centroid falls in its own bay rather than in the object.

    This is the shape the straight-line target is wrong about, and it is why the mode is a
    parameter rather than a decision taken once in the code.
    """
    rows, columns = np.indices((size, size))
    middle = size // 2
    radius = np.hypot(rows - middle, columns - middle)
    ring = (radius >= middle // 2) & (radius <= middle - 1)
    ring[middle - 4 : middle + 5, middle + 2 :] = False
    return ring.astype(np.int32)


# ------------------------------------------------------------------ the shape of the target


def test_the_two_channels_are_row_then_column_in_numpy_s_own_axis_order():
    truth = np.zeros((9, 9), dtype=np.int32)
    truth[1:8, 4] = 1  # a vertical bar: every arrow points along rows and none along columns
    flow = flows_from_instances(truth)
    assert flow.shape == (2, 9, 9)
    assert flow.dtype == np.float32
    assert np.abs(flow[0]).sum() > 0, "a bar along the rows has to move pixels along the rows"
    assert np.abs(flow[1]).max() == 0.0, "and not one pixel of it along the columns"


def test_the_field_is_unit_length_on_the_object_and_nothing_off_it():
    truth = touching_pair()
    flow = flows_from_instances(truth)
    length = np.hypot(flow[0], flow[1])
    assert length[truth == 0].max() == 0.0, "the background has no vector, not a short one"
    moving = (truth > 0) & (length > 0)
    assert np.allclose(length[moving], 1.0, atol=1e-6)


@pytest.mark.parametrize("mode", ["centroid", "diffusion"])
def test_the_field_is_exactly_zero_at_every_sink_which_is_why_no_mask_can_be_read_off_it(mode):
    """The reason `#209`'s mask comes from the loader and never from the target.

    The non-zero voxels of a flow field are the foreground *minus precisely one pixel per
    object* - the sinks - and `#211`'s oracle measured the seeds as what limits this whole
    family of methods. A mask inferred this way would be wrong exactly where the method lives
    and right everywhere a test would casually look.

    **Parametrised because it passed for the wrong reason first.** With the default mode only,
    it was asserting a property of the straight-line field, where the sink's vector is zero
    arithmetically. The diffusion's was `(0.707, 0.707)` - a full unit vector, because the
    normalisation amplifies the residue a central difference leaves across a maximum - and no
    test looked. §4's "a fixture that never contained the case".
    """
    truth = touching_pair()
    flow = flows_from_instances(truth, mode=mode)
    moving = np.hypot(flow[0], flow[1]) > 0
    assert moving.sum() == int((truth > 0).sum()) - 2, "two objects, two sinks, two zeros"
    for value in (1, 2):
        sink = _sink(truth == value)
        assert truth[sink] == value, "and the sink belongs to the object it is the sink of"
        assert not moving[sink]


def test_the_sink_is_a_pixel_of_the_object_even_when_the_centroid_is_not():
    truth = crescent()
    centre = np.argwhere(truth > 0).mean(axis=0).round().astype(int)
    assert truth[centre[0], centre[1]] == 0, "the premise: this centroid is outside the object"
    assert truth[_sink(truth > 0)] == 1


def test_an_empty_label_map_is_an_empty_field_rather_than_an_error():
    flow = flows_from_instances(np.zeros((5, 5), dtype=np.int32))
    assert flow.shape == (2, 5, 5)
    assert not flow.any()


# ------------------------------------------------------------- the two modes, as one variable


@pytest.mark.parametrize("mode", ["centroid", "diffusion"])
def test_a_touching_pair_comes_apart_though_connected_components_cannot(mode):
    truth = touching_pair()
    assert instances_from_mask(truth > 0).max() == 1, "the premise: these touch"
    flow = flows_from_instances(truth, mode=mode)
    recovered = instances_by_flows(flow, truth > 0, steps=50)
    assert recovered.max() == 2
    assert average_precision(iou_matrix(truth, recovered)) == pytest.approx(1.0)


def test_the_straight_line_shatters_a_crescent_and_the_diffusion_does_not():
    """Both numbers written down, because this is the measurement that justifies two modes.

    The straight path to the sink leaves the object, so the arrows on opposite arms of the C
    point across the bay at each other and the basins come apart. Heat cannot cross a boundary
    the object does not have, so the diffusion field runs *along* the arm.
    """
    truth = crescent()
    straight = instances_by_flows(flows_from_instances(truth, mode="centroid"), truth > 0)
    diffused = instances_by_flows(flows_from_instances(truth, mode="diffusion"), truth > 0)
    assert straight.max() >= 3, "one object, at least three basins"
    assert average_precision(iou_matrix(truth, straight)) < 0.1
    assert diffused.max() == 1
    assert average_precision(iou_matrix(truth, diffused)) == pytest.approx(1.0)


def test_the_diffusion_runs_along_the_object_and_the_straight_line_does_not():
    """The confinement measured on the field itself rather than through the decoding.

    `test_the_straight_line_shatters_a_crescent...` is an end-to-end test and it survives a
    build whose heat is **not** confined to the object - measured, by mutating the mask out of
    the diffusion and watching the crescent still come back as one object, because a pixel
    that steps into the bay lands on a zero vector and halts in a cluster the sink's own
    cluster reaches. So the confinement needs its own test and this is it: the angle between
    each vector and the straight line to the sink.

    By construction `centroid` is at 0 degrees everywhere - that is the control. `diffusion`
    puts **42 of 141** foreground pixels more than 45 degrees away from it, which is the
    field running along the arm instead of across the bay.
    """
    truth = crescent()
    mask = truth > 0
    sink = np.array(_sink(mask), dtype=float)
    away = {}
    for mode in ("centroid", "diffusion"):
        flow = flows_from_instances(truth, mode=mode)
        angles = []
        for row, column in np.argwhere(mask):
            vector = np.array([flow[0, row, column], flow[1, row, column]])
            straight = sink - (row, column)
            if np.hypot(*vector) == 0 or np.hypot(*straight) == 0:
                continue
            straight = straight / np.hypot(*straight)
            angles.append(np.degrees(np.arccos(np.clip(vector @ straight, -1.0, 1.0))))
        away[mode] = int((np.array(angles) > 45).sum())
    assert away["centroid"] == 0, "the straight line is the straight line, by construction"
    assert away["diffusion"] > 30, "and the diffusion departs from it along the arms"


def test_a_mode_that_does_not_exist_is_refused_by_name():
    with pytest.raises(InstanceError, match="one of"):
        flows_from_instances(touching_pair(), mode="heat")


def test_a_volume_is_refused_rather_than_guessed_at():
    with pytest.raises(InstanceError, match="2-D label map"):
        flows_from_instances(np.zeros((3, 4, 4), dtype=np.int32))


# ------------------------------------------------------------------------- following the field


def test_following_the_field_lands_every_pixel_on_its_own_object_s_sink():
    truth = touching_pair()
    flow = flows_from_instances(truth, mode="diffusion")
    arrived = follow_flows(flow, truth > 0, steps=100)
    rows, columns = np.nonzero(truth > 0)
    for index, (row, column) in enumerate(zip(rows, columns, strict=True)):
        sink = _sink(truth == truth[row, column])
        assert abs(arrived[0, index] - sink[0]) <= 1.0
        assert abs(arrived[1, index] - sink[1]) <= 1.0


def test_the_step_count_is_not_where_the_risk_is():
    """One step and four hundred give the same objects, and that is measured rather than hoped.

    This was written the other way round first - *too few steps labels the foreground rather
    than its basins* - and the fixture refused it. The reason is structural: the field is
    continuous, so one object's pixels land in a connected set however far along they get, and
    it **diverges** at a seam, so the gap between two objects opens on the very first step.
    Under-stepping therefore preserves the separation instead of losing it.

    Measured on fifteen of the held-out cases, 633 objects: AP 0.9914 at one step against
    0.9923 converged, a span of 0.0009 - below the 0.0013 run-to-run spread of a training run,
    so the step count cannot be read off a score at all. One step, five, ten, twenty-five,
    fifty, a hundred, two hundred and four hundred were all measured. On an exact field; a
    predicted one is a different question and only a trained run answers it.
    """
    truth = touching_pair()
    flow = flows_from_instances(truth, mode="diffusion")
    hurried = instances_by_flows(flow, truth > 0, steps=1)
    converged = instances_by_flows(flow, truth > 0, steps=400)
    assert np.array_equal(hurried, converged)
    assert average_precision(iou_matrix(truth, converged)) == pytest.approx(1.0)


def test_a_step_longer_than_a_pixel_overshoots_and_shatters_the_objects():
    """The decode parameter that *is* dangerous, with both ends of it written down.

    The field is defined on a one-pixel grid, so a step of more than a pixel jumps past the
    sink and keeps jumping; the endpoints scatter instead of converging and one object comes
    back as several. Measured on fifteen held-out cases with 633 objects: **0.9923 at a step
    of 1.0 or below, 0.4715 at 2.0 with 868 objects back, and 0.0303 at 10.0 with 4337** - so
    this is a cliff rather than a slope, and the default belongs at the bottom of it.

    On this fixture the objects are five pixels across, so the cliff sits one step further
    out; the two numbers asserted are its ends on *this* picture.
    """
    truth = touching_pair()
    flow = flows_from_instances(truth, mode="diffusion")
    exact = instances_by_flows(flow, truth > 0, steps=50, step_size=1.0)
    assert average_precision(iou_matrix(truth, exact)) == pytest.approx(1.0)

    overshot = instances_by_flows(flow, truth > 0, steps=10, step_size=5.0)
    assert overshot.max() > 2, "a step five pixels long cannot find a sink in a five-pixel block"
    assert average_precision(iou_matrix(truth, overshot)) == 0.0


def test_a_field_followed_for_no_steps_is_refused_rather_than_labelling_the_foreground():
    flow = flows_from_instances(touching_pair())
    with pytest.raises(InstanceError, match="no steps"):
        follow_flows(flow, touching_pair() > 0, steps=0)
    with pytest.raises(InstanceError, match="somewhere"):
        follow_flows(flow, touching_pair() > 0, step_size=0.0)


def test_a_field_and_a_foreground_are_two_readings_of_one_picture():
    flow = flows_from_instances(touching_pair())
    with pytest.raises(InstanceError, match="must be the same size"):
        follow_flows(flow, np.ones((4, 4), dtype=bool))
    with pytest.raises(InstanceError, match=r"\(2, H, W\)"):
        follow_flows(np.zeros((9, 11)), touching_pair() > 0)


def test_a_pixel_the_field_says_nothing_about_halts_instead_of_wandering():
    """Interpolating a field that is zero off the foreground is what stops a stray pixel."""
    truth = touching_pair()
    flow = flows_from_instances(truth)
    stray = np.zeros(truth.shape, dtype=bool)
    stray[0, 0] = True  # a corner the field has no vector for
    arrived = follow_flows(flow, stray, steps=100)
    assert arrived[0, 0] == 0.0 and arrived[1, 0] == 0.0


# ------------------------------------------------------------------------ against the rim route


def test_an_object_too_thin_to_have_an_interior_needs_no_rescue_here():
    """The structural claim of `#212`, as a test rather than as prose.

    `parts_from_instances` has a fallback for an object that erodes to nothing, because the
    rim route spends pixels on being a boundary and a thin object has none to spare. A flow
    field spends none, so a one-pixel line is an ordinary object with a sink in it.
    """
    truth = np.zeros((7, 11), dtype=np.int32)
    truth[3, 1:5] = 1
    truth[3, 7:11] = 2
    interior, _ = parts_from_instances(truth)
    assert interior.sum() == int((truth > 0).sum()), "the premise: the erosion had to rescue it"

    flow = flows_from_instances(truth, mode="diffusion")
    recovered = instances_by_flows(flow, truth > 0, steps=50)
    assert recovered.max() == 2
    assert average_precision(iou_matrix(truth, recovered)) == pytest.approx(1.0)


def test_a_minimum_area_drops_basins_and_relabels_consecutively():
    truth = np.zeros((9, 15), dtype=np.int32)
    truth[2:8, 1:7] = 1  # 36 pixels
    truth[4, 12] = 2  # one pixel
    flow = flows_from_instances(truth, mode="diffusion")
    kept = instances_by_flows(flow, truth > 0, steps=50, minimum_area=10)
    assert sorted(np.unique(kept)) == [0, 1]
    assert int((kept == 1).sum()) == 36
