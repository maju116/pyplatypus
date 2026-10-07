"""The drawing decisions, pinned - because they are values somebody will find inconvenient.

Every one of these lived only in the R package before this module, so every one of them is a
convention two packages now have to agree about. A test that only checked they exist would
let a tidy-up change what red means, and nothing downstream would fail: a wrong colour is
still a colour and a wrong alpha still renders.

The R package's suite holds its own defaults to these same values, which is the other half.
Neither repository can read the other, so each writes the numbers out - the arrangement the
reference groups and the README section order already use.
"""

from __future__ import annotations

import pyplatypus
from pyplatypus import style


def test_the_agreement_colours_are_three_and_distinct():
    """One colour would hide which of two different mistakes you are looking at."""
    colours = style.AGREEMENT_COLOURS

    assert set(colours) == {"hit", "missed", "false_alarm"}
    assert len(set(colours.values())) == 3, "two of the three are the same colour"
    assert colours == {"hit": "#3CDC5A", "missed": "#E63C3C", "false_alarm": "#F0C83C"}


def test_the_box_colours_are_the_colourblind_safe_pair():
    """ColorBrewer Dark2. Changing them to something prettier is the thing this prevents."""
    assert style.BOX_COLOURS == {"prediction": "#d95f02", "truth": "#1b9e77"}


def test_the_overlay_alpha_lets_the_image_through():
    """0.55, and the point is that both the mask and the tissue under it are visible.

    Pinned rather than range-checked: a range would permit 0.95, which renders and hides the
    boundary the reader is trying to judge.
    """
    assert style.OVERLAY_ALPHA == 0.55


def test_the_box_label_prints_two_decimals():
    """Two, because the third digit of a detector's confidence is noise at this data size."""
    rendered = style.BOX_LABEL_FORMAT.format(label="WBC", score=0.98765)

    assert rendered == "WBC 0.99"
    assert style.BOX_LABEL_FORMAT.format(label="RBC", score=1.0) == "RBC 1.00"


def test_the_drawing_threshold_is_not_the_scoring_one():
    """A picture is read by a person; `operating_point` exists for integrating a ranking.

    Drawing at the scoring threshold fills a frame with boxes the model barely considered,
    which is the same distinction `crops()` makes.
    """
    assert style.BOX_MIN_SCORE == 0.5


def test_drawing_style_carries_every_decision():
    """A caller that reads the mapping must not miss one that reads the constants.

    The R package takes this mapping across the bridge in one call, so a decision added to
    the module and not to the mapping would be invisible on that side.
    """
    carried = pyplatypus.drawing_style()

    assert carried == {
        "agreement_colours": style.AGREEMENT_COLOURS,
        "box_colours": style.BOX_COLOURS,
        "class_colours": style.CLASS_COLOURS,
        "overlay_alpha": style.OVERLAY_ALPHA,
        "box_label_format": style.BOX_LABEL_FORMAT,
        "box_min_score": style.BOX_MIN_SCORE,
    }

    constants = {name for name in vars(style) if name.isupper() and not name.startswith("_")}
    assert len(constants) == len(carried), (
        f"the module has {sorted(constants)} and the mapping carries {sorted(carried)}"
    )


def test_the_mapping_hands_out_copies():
    """A caller that mutates what it got must not change what the next caller sees."""
    first = pyplatypus.drawing_style()
    first["agreement_colours"]["hit"] = "#000000"

    assert pyplatypus.drawing_style()["agreement_colours"]["hit"] == "#3CDC5A"
