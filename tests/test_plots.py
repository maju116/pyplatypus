"""The five drawings, and the mistakes a picture does not report.

A wrong figure is the worst kind of wrong output: it renders, it looks like a result, and
nothing downstream fails. So these check correspondence rather than shape wherever they can -
a mask drawn over the image it was not computed from has the right dimensions and is still in
the wrong place, which is the defect that once made a model look unable to learn.
"""

from __future__ import annotations

import warnings

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


class _PartialDetector(_FakeDetector):
    """A split that is missing the first of three classes, which is ordinary in real data."""

    def box_shapes(self, model_name=None, split="train"):
        report = super().box_shapes(model_name, split)
        report["classes"] = ["RBC", "WBC", "Platelets"]
        report["boxes"] = {
            "width": [0.3, 0.4],
            "height": [0.3, 0.4],
            "label": [1, 2],
            "name": ["WBC", "Platelets"],
        }
        return report


def test_a_class_missing_from_a_split_does_not_recolour_the_others():
    """The colour has to mean the same thing in two figures from one model.

    Taken from a class's position among the ones *drawn*, a split without red cells would
    hand white cells the colour red cells have everywhere else - so the same cell is green
    in one figure and purple in the next, from one model, with nothing to say so. Taken
    from its position in the specification, it cannot.
    """
    full = pyplatypus.plot_anchors(_FakeDetector())
    partial = pyplatypus.plot_anchors(_PartialDetector())

    def colour_of(figure, label):
        for collection in figure.axes[0].collections:
            if collection.get_label() == label:
                return tuple(collection.get_facecolor()[0])
        raise AssertionError(f"no cloud labelled {label}")

    assert colour_of(full, "WBC") == colour_of(partial, "WBC")
    assert [c.get_label() for c in partial.axes[0].collections][:2] == ["WBC", "Platelets"]


# ------------------------------------------------------------------- instances (#214)
#
# A classes figure and an instances figure are different problems, and these tests are
# mostly about the claim that makes the second one possible: a colour may repeat across
# the picture but never between neighbours. That claim is checkable, so it is checked
# rather than promised - on grids built to make it hard, and on every pair.


def grid_of_instances(rows: int = 2, columns: int = 8, block: int = 3) -> np.ndarray:
    """A chequerboard of touching blocks, numbered so a cycling palette is sure to fail.

    Real nuclei average 0.37 neighbours each and never need more than three colours. This is
    deliberately worse than real, and **eight columns wide on purpose**: in a grid that wide
    the object above any other differs from it by exactly 8, so colouring by value modulo the
    eight-entry palette gives every vertical pair one colour. A narrower grid passes the
    control by accident - measured, on a 3x3 version where the two values that collide under
    the modulo are opposite corners and never touch, so the control did not fire and the test
    proved nothing about the choice it exists to justify.
    """
    labels = np.zeros((rows * block, columns * block), dtype=int)
    identifier = 0
    for row in range(rows):
        for column in range(columns):
            identifier += 1
            labels[row * block : (row + 1) * block, column * block : (column + 1) * block] = (
                identifier
            )
    return labels


def picture(height: int, width: int) -> np.ndarray:
    """A grey image of a given shape, since the instance maps here are not square."""
    return np.full((height, width, 3), 0.5)


def touching_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    """Every pair of object values within one step of each other, diagonals included."""
    pairs = set()
    height, width = labels.shape
    for row in range(height):
        for column in range(width):
            here = int(labels[row, column])
            if here == 0:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r, c = row + dr, column + dc
                    if 0 <= r < height and 0 <= c < width:
                        other = int(labels[r, c])
                        if other and other != here:
                            pairs.add((min(here, other), max(here, other)))
    return pairs


def test_no_two_touching_objects_are_drawn_in_one_colour():
    """The rule the figure rests on, checked on every adjacent pair rather than sampled.

    Measured over all 670 Data Science Bowl cases, 5436 adjacent pairs: an exact colouring
    needs at most 3 of the palette's 8 slots. The alternative on `#214` was a cycling fill
    whose collisions are "rare rather than impossible" - on those same pairs it paints
    **550 of 5436, 10.12%**, of touching pairs alike, which is the control below.
    """
    labels = grid_of_instances()
    pairs = touching_pairs(labels)
    assert len(pairs) >= 20, f"the premise: this grid has to be hard, got {len(pairs)} pairs"

    drawn = pyplatypus.overlay_instances(picture(*labels.shape), labels, alpha=1.0)
    centres = {
        value: tuple(
            int(c) for c in drawn[tuple(np.argwhere(labels == value).mean(axis=0).astype(int))]
        )
        for value in np.unique(labels[labels > 0])
    }
    for one, other in pairs:
        assert centres[one] != centres[other], f"objects {one} and {other} touch and match"

    # The control: colouring by value modulo the palette, which is what was rejected.
    palette = style.CLASS_COLOURS
    collisions = [(u, v) for u, v in pairs if u % len(palette) == v % len(palette)]
    assert collisions, (
        "the control did not fire, so this grid cannot tell the two schemes apart and the "
        "assertions above prove nothing about the choice"
    )


def test_an_outline_leaves_the_object_s_interior_showing():
    labels = np.zeros((9, 9), dtype=int)
    labels[2:7, 2:7] = 1
    before = picture(9, 9)
    outlined = pyplatypus.overlay_instances(before, labels, outline=True)
    filled = pyplatypus.overlay_instances(before, labels)

    middle = (4, 4)
    edge = (2, 4)
    assert tuple(outlined[middle]) == tuple(np.round(before[middle] * 255).astype(int)), (
        "an outline must not touch the interior, which is the whole reason to want one"
    )
    assert tuple(outlined[edge]) != tuple(outlined[middle]), "and it must touch the border"
    assert tuple(filled[middle]) != tuple(outlined[middle]), "a fill does cover the interior"


def test_a_one_hot_array_is_refused_because_it_cannot_carry_an_identity():
    onehot = np.zeros((8, 8, 3))
    with pytest.raises(PlatypusError, match="one integer per pixel"):
        pyplatypus.overlay_instances(image(), onehot)


def test_an_empty_palette_is_refused_rather_than_drawing_nothing():
    with pytest.raises(PlatypusError, match="nothing to draw"):
        pyplatypus.overlay_instances(image(), np.zeros((8, 8), dtype=int), colours=[])


def test_the_instance_agreement_figure_separates_found_missed_and_invented():
    """And it matches whole objects, which is the unit a pixel figure cannot show."""
    truth = np.zeros((5, 11), dtype=int)
    truth[1:4, 1:4] = 1
    truth[1:4, 5:8] = 2
    predicted = np.zeros((5, 11), dtype=int)
    predicted[1:4, 1:4] = 1
    predicted[1:4, 8:11] = 2

    shown = pyplatypus.overlay_instance_agreement(picture(5, 11), predicted, truth)
    hit, missed, invented = shown[2, 2], shown[2, 6], shown[2, 9]
    assert (hit != missed).any() and (missed != invented).any() and (hit != invented).any()


def test_a_merged_pair_reads_as_two_misses_rather_than_as_nearly_right():
    """The figure `#214` is for: a merge is almost perfect pixelwise and wrong as objects.

    One prediction covering two true objects can match at most one of them at any threshold,
    so neither true object is matched at 0.5 - and the pixel figure shows the same thing as a
    near-total hit.

    The hit colour is taken from a figure where everything *is* a hit rather than computed
    from the palette and the alpha: reproducing the blend here would make the test a copy of
    the code it checks, and a copy agrees with a wrong original.
    """
    truth = np.zeros((5, 9), dtype=int)
    truth[1:4, 1:4] = 1
    truth[1:4, 4:7] = 2
    merged = np.zeros((5, 9), dtype=int)
    merged[1:4, 1:7] = 1

    everything_matched = pyplatypus.overlay_instance_agreement(picture(5, 9), truth, truth)
    hit_colour = everything_matched[2, 2]
    shown = pyplatypus.overlay_instance_agreement(picture(5, 9), merged, truth)
    assert not (shown.reshape(-1, 3) == hit_colour).all(axis=1).any(), (
        "a merged pair must not be drawn as a hit anywhere in the figure"
    )

    pixels = pyplatypus.overlay_agreement(picture(5, 9), merged, truth)
    assert (pixels[2, 2] == pixels[2, 5]).all(), (
        "the premise: the pixel figure cannot tell this apart, which is why the object one "
        "has to exist"
    )


def test_two_instance_maps_of_different_shapes_are_refused():
    with pytest.raises(PlatypusError, match="different shapes"):
        pyplatypus.overlay_instance_agreement(
            picture(5, 5), np.zeros((5, 5), dtype=int), np.zeros((4, 4), dtype=int)
        )


def test_a_flow_field_reads_as_hue_for_direction_and_value_for_length():
    field = np.zeros((2, 1, 4))
    field[0, 0, 0] = 1.0  # down
    field[1, 0, 1] = 1.0  # right
    field[0, 0, 2] = -1.0  # up
    shown = pyplatypus.flow_to_colours(field)

    assert shown.shape == (1, 4, 3)
    hues = {tuple(int(c) for c in shown[0, index]) for index in range(3)}
    assert len(hues) == 3, f"three directions, three colours, got {hues}"
    assert tuple(int(c) for c in shown[0, 3]) == (0, 0, 0), "no vector, no colour"


def test_a_reversed_field_is_obvious_here_and_invisible_in_any_loss():
    """Why this function exists at all: `#210`'s silent failure, made visible."""
    field = np.zeros((2, 4, 4))
    field[0] = 1.0
    assert (pyplatypus.flow_to_colours(field) != pyplatypus.flow_to_colours(-field)).any()


def test_a_field_that_is_not_two_channels_is_refused():
    with pytest.raises(PlatypusError, match="2, height, width"):
        pyplatypus.flow_to_colours(np.zeros((3, 4, 4)))


def test_the_instance_figure_matches_at_the_same_strictness_as_the_metric():
    """The one way this figure can lie, pinned at the boundary where it did.

    `average_precision` compares `iou > threshold` and says why in its own module: at
    exactly 0.5, `>=` would admit two predictions each covering half an object. A merged
    prediction over two equal true objects scores exactly 0.5 against both, so it is the
    case that separates the two readings - and the drawing used the looser one until this
    test was written.
    """
    from pyplatypus.instances import average_precision, iou_matrix

    truth = np.zeros((5, 9), dtype=int)
    truth[1:4, 1:4] = 1
    truth[1:4, 4:7] = 2
    merged = np.zeros((5, 9), dtype=int)
    merged[1:4, 1:7] = 1

    overlaps = iou_matrix(truth, merged)
    assert np.allclose(overlaps, 0.5), f"the premise: this has to sit on the boundary, {overlaps}"
    assert average_precision(overlaps, [0.5]) == 0.0, "the metric counts no hit here"

    everything_matched = pyplatypus.overlay_instance_agreement(picture(5, 9), truth, truth)
    shown = pyplatypus.overlay_instance_agreement(picture(5, 9), merged, truth)
    assert not (shown.reshape(-1, 3) == everything_matched[2, 2]).all(axis=1).any(), (
        "and so must the figure, or the picture and the number disagree at 0.5"
    )


def test_a_threshold_the_metric_refuses_is_refused_here_too():
    """Below 0.5 the match stops being one to one, so the figure would draw a rule the
    number does not use. `average_precision` refuses it; this refuses it in the same words'
    worth of reason."""
    truth = np.zeros((5, 9), dtype=int)
    truth[1:4, 1:4] = 1
    with pytest.raises(PlatypusError, match="below 0.5"):
        pyplatypus.overlay_instance_agreement(picture(5, 9), truth, truth, threshold=0.4)


def three_mutually_touching() -> np.ndarray:
    """Three objects each touching the other two, so two colours provably cannot do it."""
    labels = np.zeros((7, 7), dtype=int)
    labels[1:4, 1:4] = 1
    labels[1:4, 4:7] = 2
    labels[4:7, 1:7] = 3
    return labels


def test_a_palette_too_short_to_colour_the_picture_says_so():
    """The wrap is the one case this figure answers its own question wrongly in - `#231`.

    `_colour_index` said `overlay_instances` reported it and nothing did: no `warnings.warn`
    anywhere in the module, for as long as the function has existed. The declaration was
    right about what should happen, which is why this is the warning and not a deletion.
    """
    labels = three_mutually_touching()

    with pytest.warns(UserWarning, match="needs 3 colours") as caught:
        pyplatypus.overlay_instances(picture(7, 7), labels, colours=style.CLASS_COLOURS[:2])

    message = str(caught[0].message)
    assert "the palette has 2" in message, message
    assert "1 of 3" in message, f"one object of three has to wrap, not: {message}"


def test_a_wrapped_object_is_drawn_in_a_neighbour_s_colour():
    """Which is the whole reason the wrap is worth a word, and it is provable.

    A slot is only past the end of the palette once every colour before it was taken by a
    neighbour, so the wrap lands on one a neighbour holds - never on a free colour. Measured
    here rather than argued, because the warning's text makes that claim to a reader.
    """
    labels = three_mutually_touching()
    with pytest.warns(UserWarning):
        drawn = pyplatypus.overlay_instances(
            picture(7, 7), labels, alpha=1.0, colours=style.CLASS_COLOURS[:2]
        )

    centres = {
        value: tuple(int(c) for c in drawn[tuple(centre)])
        for value, centre in ((1, (2, 2)), (2, (2, 5)), (3, (5, 3)))
    }
    alike = [
        (one, other) for one, other in touching_pairs(labels) if centres[one] == centres[other]
    ]
    assert alike, f"the wrap has to collide with a neighbour, and nothing did: {centres}"


def test_the_number_of_colours_the_warning_asks_for_is_enough():
    """A warning naming a number that does not fix it would be worse than silence."""
    labels = three_mutually_touching()
    with pytest.warns(UserWarning, match="needs 3 colours"):
        pyplatypus.overlay_instances(picture(7, 7), labels, colours=style.CLASS_COLOURS[:2])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pyplatypus.overlay_instances(picture(7, 7), labels, colours=style.CLASS_COLOURS[:3])


def test_a_picture_the_palette_can_hold_warns_about_nothing():
    """The control: a grid built to be harder than real data, and the default palette.

    Without this the test above would pass on a function that warns on every call, which is
    the calibration half of §4's lesson about checks that admit what they were written to
    prevent - and a figure that cries wolf teaches a reader to ignore the one case that
    means something.
    """
    labels = grid_of_instances()
    assert len(touching_pairs(labels)) >= 20, "the premise: this grid has to be hard"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pyplatypus.overlay_instances(picture(*labels.shape), labels)


def test_the_agreement_overlay_colours_a_gapped_map_by_its_own_object_numbers():
    """`pyplatypus#325`'s own regression, and the engine had no test that could see it.

    `iou_matrix`'s rows became the object numbers **present** rather than `1..max`, and this
    function turned a row back into a label with `row + 1` - correct only while the numbering
    was consecutive. `0.8.0a55` shipped the first half and not the second, so an agreement
    figure over a gapped map coloured the wrong objects.

    What caught it was `rplatypus`'s drawing pair, which compares R's overlay with this one
    pixel for pixel and carries a fixture named `gapped`. That is §3F working in the
    direction it was built for - and it means this suite was blind, which is what this test
    is for.
    """
    from pyplatypus.plots import overlay_instance_agreement

    truth = np.zeros((8, 8), dtype=np.int32)
    truth[1:4, 1:4] = 1
    truth[5:8, 5:8] = 4  # a gap: the labels present are 1 and 4
    assert sorted(np.unique(truth)) == [0, 1, 4], "no gap, so this proves nothing"

    picture = np.full((8, 8, 3), 128.0)
    both = overlay_instance_agreement(picture.copy(), truth, truth.copy(), threshold=0.5)
    # Everything matched, so every object is drawn as a hit and the two object regions carry
    # the same colour - which is the assertion `row + 1` failed: it marked object 2 as
    # matched, object 4 as missed, and coloured them differently.
    first = both[2, 2].tolist()
    second = both[6, 6].tolist()
    assert first == second, f"the two matched objects are coloured differently: {first} {second}"
    assert first != [128, 128, 128], "nothing was drawn, so the comparison above is vacuous"

    # And the control in the other direction: an object that genuinely is not matched gets a
    # different colour from one that is.
    predicted = np.zeros((8, 8), dtype=np.int32)
    predicted[1:4, 1:4] = 7  # matches truth's object 1, and nothing matches truth's 4
    partly = overlay_instance_agreement(picture.copy(), truth, predicted, threshold=0.5)
    assert partly[2, 2].tolist() != partly[6, 6].tolist()
