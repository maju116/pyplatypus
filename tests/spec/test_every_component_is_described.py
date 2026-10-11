"""Every component you can name in a configuration says what it is for.

The field-level half of this is `test_every_field_is_described.py`: all 201 described
properties, down to `eps`. This is the level above it, and it was the one missing. Of the 29
components - 10 losses, 4 metrics, 8 optimisers, 7 callbacks - **three** carried a
description: `boundary`, `cosine_annealing` and `swa`, the three written most recently with
a measurement in hand. `smooth` explained itself and `lovasz` did not say what a Lovasz loss
is for, so the generated configuration reference printed a heading and a table of knobs with
nothing under the heading, and `?losses` in R listed ten names without the one sentence a
reader actually needs: *which of these do I want*.

**Driven from the schema**, for the reason the field test gives at length: a walk over the
models by hand reported 100% at 87% because it never unwrapped `Annotated`. A component is a
definition whose `name` is a literal - that is what makes it one of a discriminated union -
which finds 29 and correctly leaves out `AugmentationStep` and `SegmentationModel`, whose
`name` is prose the user writes.
"""

import pytest

from pyplatypus import spec_schema

SCHEMA = spec_schema()
DEFS = SCHEMA["$defs"]
COMPONENTS = sorted(
    (model, definition.get("description") or "")
    for model, definition in DEFS.items()
    if "const" in (definition.get("properties", {}).get("name") or {})
)


def test_the_search_finds_every_component():
    """Or this file passes by looking at four of them."""
    assert len(COMPONENTS) == 29
    names = {model for model, _ in COMPONENTS}
    # one of each kind, so a union dropping out of the schema fails here rather than later
    for expected in ("DiceLoss", "ClDiceMetric", "Sgd", "TerminateOnNaN"):
        assert expected in names
    # and the two whose `name` is a free string are not components
    assert "AugmentationStep" not in names
    assert "SegmentationModel" not in names


@pytest.mark.parametrize(
    "model", [model for model, _ in COMPONENTS], ids=[model for model, _ in COMPONENTS]
)
def test_every_component_has_a_description(model):
    description = next(d for m, d in COMPONENTS if m == model)
    assert description, (
        f"{model} has no description, so its heading in the configuration reference is "
        f"empty and `available_losses()` and friends have nothing to print. The docstring "
        f"of the pydantic model is what the schema publishes."
    )


def test_a_component_description_answers_which_one_to_use():
    """A floor of 150 characters against the field test's 30, because the question here is
    larger. `IouMetric` is the shortest at 270 and it still has room to say that it ranks
    models identically to Dice - a phrase cannot do that, and a phrase is what this level
    had three of."""
    thin = [f"{model} ({len(d)})" for model, d in COMPONENTS if len(d) < 150]
    assert thin == []


def test_no_description_merely_restates_the_name():
    """The failure this level invites: "Dice loss." under the heading `dice`. Checked by
    looking for any sentence of substance beyond the first mention of the component's own
    literal - crude, but it is the shape the three that existed were written against."""
    restatements = []
    for model, description in COMPONENTS:
        literal = DEFS[model]["properties"]["name"]["const"]
        without_name = (
            description.lower().replace(literal.replace("_", " "), "").replace(literal, "")
        )
        if len(without_name) < 100:
            restatements.append(model)
    assert restatements == []
