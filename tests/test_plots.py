"""The five drawings, and the mistakes a picture does not report.

A wrong figure is the worst kind of wrong output: it renders, it looks like a result, and
nothing downstream fails. So these check correspondence rather than shape wherever they can -
a mask drawn over the image it was not computed from has the right dimensions and is still in
the wrong place, which is the defect that once made a model look unable to learn.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

import pyplatypus
from pyplatypus import PlatypusError, style

COLORMAP = [(0, 0, 0), (255, 0, 0)]


def image(size: int = 8) -> np.ndarray:
    return np.full((size, size, 3), 0.5)


def mask(size: int = 8, rows: slice = slice(2, 5), cols: slice = slice(2, 5)) -> np.ndarray:
    onehot = np.zeros((size, size, 2))
    onehot[:, :, 0] = 1
    onehot[rows, cols, 0] = 0
    onehot[rows, cols, 1] = 1
    return onehot


def test_an_overlay_tints_the_mask_and_leaves_the_background():
    """The point of an overlay is that both are visible; tinting everything shows nothing."""
    shown = pyplatypus.overlay_mask(image(), mask(), COLORMAP)

    assert shown.shape == (8, 8, 3)
    assert shown.dtype == np.uint8
    assert (shown[3, 3] != shown[0, 0]).any(), "the mask was not drawn"
    np.testing.assert_array_equal(shown[0, 0], shown[7, 7])
    assert shown[0, 0].tolist() == [128, 128, 128], "the background was tinted"


def test_the_alpha_is_the_engines_and_changes_what_is_drawn():
    """A default that does not reach the arithmetic is a default in name only."""
    faint = pyplatypus.overlay_mask(image(), mask(), COLORMAP, alpha=0.1)
    strong = pyplatypus.overlay_mask(image(), mask(), COLORMAP, alpha=0.9)
    default = pyplatypus.overlay_mask(image(), mask(), COLORMAP)

    assert faint[3, 3][0] < default[3, 3][0] < strong[3, 3][0]
    np.testing.assert_array_equal(
        default, pyplatypus.overlay_mask(image(), mask(), COLORMAP, alpha=style.OVERLAY_ALPHA)
    )


def test_a_mask_that_does_not_cover_the_image_is_refused():
    """Right shape, wrong place is the failure a picture cannot report by itself."""
    with pytest.raises(PlatypusError, match="wrong place"):
        pyplatypus.overlay_mask(image(8), mask(6), COLORMAP)


def test_agreement_draws_three_different_things_in_three_colours():
    """A single colour would hide which of two different mistakes is being looked at."""
    predicted = np.zeros((8, 8), dtype=int)
    predicted[2:5, 2:5] = 1
    actual = np.zeros((8, 8), dtype=int)
    actual[3:6, 2:5] = 1

    shown = pyplatypus.overlay_agreement(image(), predicted, actual)

    hit, invented, missed = shown[3, 3], shown[2, 3], shown[5, 3]
    assert len({tuple(hit), tuple(invented), tuple(missed)}) == 3
    np.testing.assert_array_equal(shown[0, 0], [128, 128, 128])


def test_agreement_without_all_three_colours_is_refused():
    with pytest.raises(PlatypusError, match="missing"):
        pyplatypus.overlay_agreement(image(), mask(), mask(), colours={"hit": "#ffffff"})


def test_plot_masks_draws_agreement_only_when_both_are_given():
    """Agreement between a prediction and nothing is not defined."""
    one = pyplatypus.plot_masks(image()[None], prediction=mask()[None], colormap=COLORMAP)
    both = pyplatypus.plot_masks(
        image()[None], prediction=mask()[None], truth=mask()[None], colormap=COLORMAP
    )

    assert isinstance(one, Figure)
    assert [axis.get_title() for axis in one.axes] == ["image", "prediction"]
    assert [axis.get_title() for axis in both.axes] == [
        "image",
        "truth",
        "prediction",
        "agreement",
    ]


def test_drawing_a_mask_without_a_colormap_is_refused():
    with pytest.raises(PlatypusError, match="colormap"):
        pyplatypus.plot_masks(image()[None], prediction=mask()[None])


def test_plot_boxes_honours_the_threshold_and_labels_what_it_draws():
    record = [
        {
            "boxes": [[1, 1, 5, 5], [2, 2, 4, 4]],
            "labels": ["cell", "cell"],
            "scores": [0.9, 0.1],
        }
    ]

    figure = pyplatypus.plot_boxes(image()[None], record, min_score=0.5)
    axis = figure.axes[0]

    assert len(axis.patches) == 1, "the box below the threshold was drawn"
    assert [text.get_text() for text in axis.texts] == ["cell 0.90"]


def test_plot_boxes_refuses_a_record_count_that_does_not_match():
    """A box drawn on the wrong frame is wrong in a way no score reports."""
    with pytest.raises(PlatypusError, match="record"):
        pyplatypus.plot_boxes(np.stack([image(), image()]), [{"boxes": [[1, 1, 2, 2]]}])


def test_plot_anchors_draws_every_group_and_the_boxes_under_them():
    figure = pyplatypus.plot_anchors(
        [[(0.1, 0.1), (0.2, 0.3)], [(0.4, 0.5)]], boxes=np.full((20, 2), 0.25)
    )
    axis = figure.axes[0]

    assert len(axis.collections) == 3, "one for the boxes and one per anchor group"
    assert axis.get_xscale() == "log", "linear axes collapse the smallest class into a corner"


def test_the_figures_are_returned_rather_than_shown_or_saved(tmp_path):
    """Never `plt.show()` and never a file: the caller decides, and this is the testable one.

    `pyplot` keeps a registry of figures it manages; using it here would leak one per call
    and need a display. A `Figure` built directly is in no registry, which is what this
    checks.
    """
    import matplotlib.pyplot

    before = matplotlib.pyplot.get_fignums()
    figure = pyplatypus.plot_masks(image()[None], prediction=mask()[None], colormap=COLORMAP)

    assert matplotlib.pyplot.get_fignums() == before, "the figure went into pyplot's registry"
    assert not list(tmp_path.iterdir()), "something was written to disk"
    assert isinstance(figure, Figure)
