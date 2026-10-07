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


def test_the_default_colormap_is_the_two_class_one_r_also_defaults_to():
    """No colormap draws the binary case rather than refusing, as `plot_masks()` does in R.

    Checked by drawing the same masks twice: once with the default and once with the list
    spelled out. Equal pictures is the assertion - a default that drew something else would
    pass any test that only asked whether a figure came back.
    """
    default = pyplatypus.plot_masks(image()[None], prediction=mask()[None])
    spelled = pyplatypus.plot_masks(
        image()[None], prediction=mask()[None], colormap=[(0, 0, 0), (255, 255, 255)]
    )
    assert np.array_equal(
        default.axes[1].get_images()[0].get_array(),
        spelled.axes[1].get_images()[0].get_array(),
    )


def test_a_class_the_colormap_has_no_colour_for_is_refused():
    """The two ways into a mask used to read this differently.

    A one-hot mask with more channels than the colormap has colours was refused by name; an
    index mask was clipped, so classes 2 and 3 were drawn in class 1's colour and nothing
    said so. One function, one answer.
    """
    indices = np.zeros((1, 8, 8), dtype=int)
    indices[0, 2, 2] = 3
    with pytest.raises(PlatypusError, match="class 3 but the colormap defines 2"):
        pyplatypus.plot_masks(image()[None], prediction=indices)

    onehot = np.zeros((1, 8, 8, 4))
    onehot[0, 2, 2, 3] = 1
    with pytest.raises(PlatypusError, match="4 channels but the colormap defines 2"):
        pyplatypus.plot_masks(image()[None], prediction=onehot)


def test_a_volume_needs_a_slice_and_an_image_refuses_one():
    """R refuses to draw a volume as one picture, and so does this.

    The two mask layouts reach the plane by different axes - one-hot carries channels, an
    index mask does not - so both are drawn here. A wrong axis would take a plane of the
    right shape out of the wrong place, which is the failure no shape assertion sees.
    """
    volumes = np.zeros((2, 8, 8, 6, 3))
    volumes[:, :, :, 3, :] = 0.5
    onehot = np.zeros((2, 8, 8, 6, 2))
    onehot[:, 2:5, 2:5, 3, 1] = 1
    indices = onehot.argmax(axis=-1)

    with pytest.raises(PlatypusError, match="volume shown as one picture"):
        pyplatypus.plot_masks(volumes, prediction=onehot)

    for prediction in (onehot, indices):
        figure = pyplatypus.plot_masks(volumes, prediction=prediction, slice="middle")
        drawn = figure.axes[1].get_images()[0].get_array()
        assert drawn.shape == (8, 8, 3)
        # The plane that holds the lesion, so a wrong axis shows background instead.
        assert (drawn[2:5, 2:5] != drawn[0, 0]).any()

    with pytest.raises(PlatypusError, match="outside 0..5"):
        pyplatypus.plot_masks(volumes, prediction=onehot, slice=9)
    with pytest.raises(PlatypusError, match="applies to volumes"):
        pyplatypus.plot_masks(image()[None], prediction=mask()[None], slice="middle")


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


def test_a_prediction_record_is_labelled_with_the_class_name():
    """`predict` returns both, and only one of them is a name.

    Its records carry the class indices under `labels` and the names under `names` - the
    same record, two keys - so reading `labels` labelled every box of every real prediction
    with an integer. The figure looked right and said `0 0.90`.
    """
    record = [
        {
            "key": "frame",
            "boxes": [[1, 1, 5, 5]],
            "scores": [0.9],
            "labels": [0],
            "names": ["RBC"],
        }
    ]

    figure = pyplatypus.plot_boxes(image()[None], record)
    assert [text.get_text() for text in figure.axes[0].texts] == ["RBC 0.90"]


def test_plot_boxes_refuses_a_record_count_that_does_not_match():
    """A box drawn on the wrong frame is wrong in a way no score reports."""
    with pytest.raises(PlatypusError, match="record"):
        pyplatypus.plot_boxes(np.stack([image(), image()]), [{"boxes": [[1, 1, 2, 2]]}])


def test_plot_anchors_draws_every_group_and_the_boxes_under_them():
    figure = pyplatypus.plot_anchors(
        [[(0.1, 0.1), (0.2, 0.3)], [(0.4, 0.5)]], boxes=np.full((20, 2), 0.25)
    )
    axis = figure.axes[0]

    assert len(axis.collections) == 2, "one for the boxes, one for every anchor"
    # Linear by default, which is what R does. The log scale is worth having and is the
    # caller's: a default that differed between the two packages would make the same call
    # draw two different pictures, which is the whole thing `style.py` exists to stop.
    assert axis.get_xscale() == "linear"
    assert pyplatypus.plot_anchors([[(0.1, 0.1)]], log=True).axes[0].get_xscale() == "log"


class _FakeDetector:
    """A stand-in for a fitted detector, answering the one call the figure makes.

    Not a trained engine: this asserts that `plot_anchors` asks `box_shapes` for both clouds
    and draws what comes back, and training a 61.5-million-parameter network to find that
    out would add ten seconds to the suite for no extra question answered. That the engine's
    own numbers are right is `test_detection_engine.py`'s job.
    """

    def __init__(self):
        self.asked = None

    def box_shapes(self, model_name=None, split="train"):
        self.asked = (model_name, split)
        return {
            "boxes": {
                "width": [0.1, 0.2, 0.3, 0.4],
                "height": [0.1, 0.2, 0.3, 0.4],
                "label": [0, 0, 1, 1],
                "name": ["RBC", "RBC", "WBC", "WBC"],
            },
            "anchors": [[[0.1, 0.1], [0.2, 0.2]], [[0.4, 0.4]]],
            "anchors_were_fitted": True,
            "input_shape": [416, 416],
            "classes": ["RBC", "WBC"],
        }


def test_an_engine_is_the_second_way_in_and_the_classes_are_separated():
    """R's `plot_anchors()` takes a fit; this takes the engine, and asks it for both clouds.

    One cloud per class, because a figure in which every box is the same grey cannot show a
    class the anchors have nothing near - which is the question the picture is drawn for.
    """
    detector = _FakeDetector()
    figure = pyplatypus.plot_anchors(detector, split="validation")
    axis = figure.axes[0]

    assert detector.asked == (None, "validation"), "no name means the first model"
    assert [collection.get_label() for collection in axis.collections] == [
        "RBC",
        "WBC",
        "anchors",
    ]
    # What the figure says about itself, so a reader knows whether the anchors were fitted
    # or declared, and in which coordinates the clouds are.
    assert "4 boxes in 'validation'" in axis.get_title()
    assert "3 anchors fitted to the training boxes" in axis.get_title()
    assert "416 x 416" in axis.get_title()


def test_an_engine_and_boxes_together_are_refused():
    """Two answers to one question: the engine has the boxes it was asked about."""
    with pytest.raises(PlatypusError, match="two answers to the same question"):
        pyplatypus.plot_anchors(_FakeDetector(), boxes=np.full((4, 2), 0.2))


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


def test_the_mask_panels_are_the_mask_and_the_agreement_panel_is_over_the_image():
    """Which column shows what, pinned - because the two packages disagreed about it.

    R draws truth and prediction as the mask in its own colours and keeps the image
    underneath only for the agreement panel. This drew all three over the image, so the same
    call made a visibly different figure on each side, which is the thing `style.py` exists
    to prevent one level down.

    Asserted on the pixels rather than on the titles: a mask panel holds nothing but
    colormap colours, and the agreement panel holds pixels of the image.
    """
    picture = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    picture[..., 0] = 200  # a red image, which neither the colormap nor the overlay uses
    marks = np.zeros((1, 8, 8), dtype=int)
    marks[0, 2:5, 2:5] = 1

    figure = pyplatypus.plot_masks(picture, prediction=marks, truth=marks)
    titles = [axis.get_title() for axis in figure.axes]
    panels = dict(zip(titles, (axis.get_images()[0].get_array() for axis in figure.axes)))

    for name in ("truth", "prediction"):
        colours = {tuple(pixel) for pixel in np.asarray(panels[name]).reshape(-1, 3)}
        assert colours == {(0, 0, 0), (255, 255, 255)}, (name, colours)

    agreement = np.asarray(panels["agreement"])
    assert (agreement[0, 0] == (200, 0, 0)).all(), "the image is not under the agreement"
    assert not (agreement[3, 3] == (200, 0, 0)).all(), "the agreement was not drawn"
