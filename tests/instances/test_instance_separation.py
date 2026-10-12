"""Interior plus rim: the decomposition, and growing it back.

The whole idea is that touching objects stop touching when each one loses its own rim, so
connected components - which cannot tell two touching nuclei apart - can tell their interiors
apart. These tests are mostly about the cases where that is not simply true: an object too
thin to have an interior, an object at the edge of the picture, an interior that falls into
two pieces, and a foreground the prediction left without any centre at all.

Measured on the Data Science Bowl's 134 held-out cases before any of this was trusted: the
ground truth through this post-process scores an instance AP of **0.9846** where the same
truth through connected components scores 0.7655, so the post-process is not what caps the
approach. The strict erosion would have erased **6 objects in 4 of those cases**, which is
why the fallback below is not hypothetical.
"""

import numpy as np
import pytest

from pyplatypus.instances import average_precision, instances_from_mask, iou_matrix
from pyplatypus.instances.metrics import InstanceError
from pyplatypus.instances.postprocess import instances_from_parts, parts_from_instances


def touching_pair(size=(9, 11)):
    """Two blocks sharing a seam, which is the case connected components cannot read."""
    truth = np.zeros(size, dtype=np.int32)
    truth[2:7, 1:6] = 1
    truth[2:7, 6:11] = 2
    return truth


# ------------------------------------------------------------------ the decomposition


def test_the_parts_are_disjoint_and_cover_the_foreground():
    truth = touching_pair()
    interior, rim = parts_from_instances(truth)
    assert not (interior & rim).any(), "a pixel cannot be interior and rim at once"
    assert np.array_equal(interior | rim, truth > 0)
    assert interior.dtype == bool and rim.dtype == bool


def test_each_object_loses_its_own_rim_so_the_interiors_come_apart():
    truth = touching_pair()
    assert instances_from_mask(truth > 0).max() == 1, "the premise: these touch"
    interior, _ = parts_from_instances(truth)
    assert instances_from_mask(interior).max() == 2, "and their interiors do not"


def test_the_pair_round_trips_the_truth_exactly():
    truth = touching_pair()
    interior, _ = parts_from_instances(truth)
    recovered = instances_from_parts(interior, truth > 0)
    assert recovered.max() == 2
    assert average_precision(iou_matrix(truth, recovered)) == pytest.approx(1.0)


def test_the_picture_s_own_edge_is_not_a_rim():
    """Padded by edge replication, not with zeros. A zero pad makes the image border look
    like background, so an object running off the edge is given a rim that separates it from
    nothing - interior spent for no gain."""
    truth = np.zeros((7, 7), dtype=np.int32)
    truth[0:4, 0:4] = 1  # flush against the top-left corner
    interior, _ = parts_from_instances(truth)
    assert interior[0, 0], "the corner pixel has no object on the other side to part from"
    assert not interior[3, 3], "but the inward edge is a real rim"


def test_an_object_too_thin_to_erode_survives_rather_than_vanishing():
    """Six objects in four of the 134 held-out cases are this thin. An object absent from
    the target is one the model is told not to find: a false negative every epoch, with
    nothing to say the target was at fault rather than the prediction."""
    truth = np.zeros((7, 9), dtype=np.int32)
    truth[3, 1:8] = 1  # one pixel wide, so a one-pixel erosion erases it
    interior, rim = parts_from_instances(truth)
    assert interior.any(), "the whole object would otherwise be rim and unfindable"
    assert np.array_equal(interior, truth > 0), "nothing touches it, so it keeps itself"
    assert not rim.any()


def test_a_thin_object_beside_another_keeps_only_what_does_not_touch_it():
    """The weaker fallback: erode where a *different object* touches, not where background
    does. Otherwise rescuing a thin object would hand it a seed shared with its neighbour."""
    truth = np.zeros((7, 9), dtype=np.int32)
    truth[3, 1:5] = 1  # one pixel wide
    truth[2:5, 5:8] = 2  # a block immediately to its right
    interior, _ = parts_from_instances(truth)
    assert interior[3, 1], "the far end of the thin object is kept"
    assert not interior[3, 4], "the end that touches object 2 is not"
    assert instances_from_mask(interior).max() == 2, "two seeds, not one shared"


def test_an_object_wedged_on_every_side_keeps_itself_whole():
    """The last resort, and it is a merge - but a merge of an object that could not be
    represented at all is the better of the two errors."""
    truth = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3],
        ],
        dtype=np.int32,
    )
    interior, _ = parts_from_instances(truth)
    assert (interior & (truth == 2)).any(), "object 2 is one pixel tall between two others"


def test_a_wider_erosion_is_accepted_and_removes_more():
    truth = np.zeros((11, 11), dtype=np.int32)
    truth[1:10, 1:10] = 1
    one, _ = parts_from_instances(truth, width=1)
    two, _ = parts_from_instances(truth, width=2)
    assert two.sum() < one.sum() < (truth > 0).sum()


def test_an_erosion_of_nothing_is_refused_rather_than_doing_nothing():
    with pytest.raises(InstanceError, match="removes no rim"):
        parts_from_instances(touching_pair(), width=0)


def test_a_volume_is_refused_rather_than_guessed_at():
    with pytest.raises(InstanceError, match="separate question"):
        parts_from_instances(np.zeros((4, 4, 4), dtype=np.int32))


# ------------------------------------------------------------------ growing them back


def test_every_foreground_pixel_joins_the_nearest_seed():
    """Not the first seed found, and not the largest: the distance decides. Two seeds at
    different distances from one strip of foreground, and the split lands at the midline."""
    interior = np.zeros((1, 11), dtype=bool)
    interior[0, 0] = True
    interior[0, 10] = True
    grown = instances_from_parts(interior, np.ones((1, 11), dtype=bool))
    assert grown[0, :5].tolist() == [1] * 5
    assert grown[0, 6:].tolist() == [2] * 5


def test_foreground_outside_the_mask_is_not_invented():
    interior = np.zeros((5, 5), dtype=bool)
    interior[2, 2] = True
    foreground = np.zeros((5, 5), dtype=bool)
    foreground[1:4, 1:4] = True
    grown = instances_from_parts(interior, foreground)
    assert grown.max() == 1
    assert not grown[foreground == 0].any(), "nothing grows where no object was predicted"


def test_a_foreground_with_no_interior_anywhere_stays_background():
    """The model said "object here" and "no centre anywhere". Inventing a centre would turn a
    contradiction into a confident wrong count."""
    grown = instances_from_parts(np.zeros((5, 5), dtype=bool), np.ones((5, 5), dtype=bool))
    assert grown.max() == 0
    assert grown.shape == (5, 5)


def test_an_interior_that_falls_into_two_pieces_becomes_two_objects():
    """Measured rather than hypothetical: this gained 52 objects over the 134 held-out cases,
    in 20 of them, and it is most of the 0.0154 the ceiling sits below 1.0. Pinned here so
    the behaviour is known rather than discovered later."""
    truth = np.zeros((9, 13), dtype=np.int32)
    truth[3:6, 1:5] = 1
    truth[3:6, 8:12] = 1  # one object, two lobes
    truth[4, 5:8] = 1  # joined by a one-pixel waist, which the erosion cuts
    interior, _ = parts_from_instances(truth)
    recovered = instances_from_parts(interior, truth > 0)
    assert truth.max() == 1
    assert recovered.max() == 2, "one object came back as two"


def test_a_minimum_area_drops_regions_and_relabels_consecutively():
    interior = np.zeros((5, 20), dtype=bool)
    interior[2, 1] = True  # will grow into a small region
    interior[2, 10:18] = True
    foreground = np.zeros((5, 20), dtype=bool)
    foreground[2, 0:2] = True
    foreground[2, 9:19] = True
    loose = instances_from_parts(interior, foreground)
    assert loose.max() == 2
    tight = instances_from_parts(interior, foreground, minimum_area=5)
    assert tight.max() == 1, "the two-pixel region goes"
    assert sorted(np.unique(tight)) == [0, 1], "and the labels stay consecutive"


def test_two_readings_of_one_picture_must_be_the_same_size():
    with pytest.raises(InstanceError, match="same size"):
        instances_from_parts(np.zeros((4, 4), dtype=bool), np.zeros((4, 5), dtype=bool))


def test_a_volume_interior_is_refused():
    with pytest.raises(InstanceError, match="2-D interior"):
        instances_from_parts(np.zeros((4, 4, 4), dtype=bool), np.zeros((4, 4, 4), dtype=bool))
