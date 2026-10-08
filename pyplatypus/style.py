"""The decisions a drawing makes, as data rather than as code in two places.

These values were the R package's first, and were lifted here when this one learnt to draw
in 0.8.0a1. A Python drawing that chose its own would make two packages that disagree about
what red means - and the arrangement that prevents it already has a precedent here: one
`_brand.yml`, two documentation sites, verified to emit a byte-identical stylesheet. Same
shape, one level down.

So these are the engine's. The R package mirrors them in `R/style.R` rather than fetching
them, because drawing there must keep working with no Python at all, and
`tests/testthat/test-drawing-style.R` compares every value against `drawing_style()` so the
mirror cannot drift. Each carries the reason its value is what it is - the same treatment the `(centre, width)` window table gets, and for the same
reason: a number without its reason gets changed by whoever finds it inconvenient.

Not specification fields. A user does not configure them in a YAML file, because they are
not about the run; they are about how its answer is shown.
"""

from __future__ import annotations

#: Found, missed, invented - three colours rather than one, because **a missed lesion and a
#: false alarm cost different things** and a single-colour overlay hides which of the two you
#: are looking at. Chosen for separability rather than for taste: green, red and amber stay
#: distinguishable in greyscale print, which a figure in a paper may become.
AGREEMENT_COLOURS: dict[str, str] = {
    "hit": "#3CDC5A",
    "missed": "#E63C3C",
    "false_alarm": "#F0C83C",
}

#: One colour per class, for a figure that separates classes rather than predictions from
#: truth. The whole of ColorBrewer's Dark2, of which `BOX_COLOURS` above is the first two
#: entries - so the palettes are one decision rather than two that happen to overlap.
#:
#: Eight, and then it repeats. A figure with more than eight classes in it cannot be read by
#: colour whatever palette it uses, and cycling says that plainly where inventing a ninth
#: colour would pretend otherwise.
#:
#: R's `plot_anchors()` used ggplot2's default hue scale until it began reading this, which
#: is why the palette is shared rather than merely matching: the default is not
#: colourblind-safe and these are.
CLASS_COLOURS: list[str] = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]

#: Prediction against truth, for boxes. The first two entries of `CLASS_COLOURS` above,
#: taken rather than restated: this comment used to claim the two palettes were one
#: decision while the source wrote the same hex codes twice, which is this module's own
#: subject. Roughly 1 in 12 men cannot separate red from green, and a figure nobody can
#: read is a figure that failed.
BOX_COLOURS: dict[str, str] = {
    "prediction": CLASS_COLOURS[1],
    "truth": CLASS_COLOURS[0],
}


#: How much of the image an overlay lets through. 0.55 keeps the tissue legible underneath
#: while the mask still reads as a region rather than a tint - and the point of an overlay is
#: that both are visible at once, or a reader cannot tell whether the boundary is right.
#:
#: It appeared four times across two R files before it lived here, which is the smaller
#: version of the problem this module exists for.
OVERLAY_ALPHA: float = 0.55

#: A box's label: the class, then its score to two decimals. Two rather than three because
#: the third digit of a detector's confidence is noise at any dataset size anyone has here -
#: average precision on BCCD moves by 0.0159 between seeds - and a figure that prints it
#: implies a precision the model does not have.
BOX_LABEL_FORMAT: str = "{label} {score:.2f}"

#: Boxes below this are not drawn. Deliberately **not** the engine's `operating_point`: that
#: one sits near zero so average precision can integrate the whole ranking, and drawing from
#: that tail fills a frame with boxes the model barely considered. A picture is read by a
#: person, so it gets the threshold a person would want.
BOX_MIN_SCORE: float = 0.5


def drawing_style() -> dict[str, object]:
    """Every decision above, in one mapping.

    One call rather than five, because the R package reaches these across the bridge and a
    round trip per constant is a round trip per constant.

    Returns:
        Every decision in this module, keyed by a lower-case name. The mappings inside are
        copies, so a caller that mutates what it got back does not change what the next
        caller sees.

    >>> style = drawing_style()
    >>> for name in sorted(style):
    ...     print(name)
    agreement_colours
    box_colours
    box_label_format
    box_min_score
    class_colours
    overlay_alpha
    >>> style["agreement_colours"]["missed"]
    '#E63C3C'
    >>> style["box_label_format"].format(label="WBC", score=0.98765)
    'WBC 0.99'
    """
    return {
        "agreement_colours": dict(AGREEMENT_COLOURS),
        "box_colours": dict(BOX_COLOURS),
        "class_colours": list(CLASS_COLOURS),
        "overlay_alpha": OVERLAY_ALPHA,
        "box_label_format": BOX_LABEL_FORMAT,
        "box_min_score": BOX_MIN_SCORE,
    }
