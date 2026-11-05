"""Every model field a builder reads, asked whether it reaches the built network.

`maju116/pyplatypus#395`, and the shape is the point rather than the list. The tenth and
eleventh review passes measured the same class twice: a field is refused by `spec/` when it
is wrong, described in the schema, given `@param` prose in R - and then **read once** by a
builder, with nothing asserting that the built object reflects it. Of 40 engine read sites
measured, 21 were held by nothing.

**So this is one enumerating test rather than one test per field.** Every survivor in both
passes was a read nobody enumerated, so a per-field test leaves the next new field in
exactly the state these two passes found - which is why `#386`'s six individual tests, good
as they are, could not have prevented this.

The question asked of each field is the weakest one that cannot be satisfied by accident:
**does a non-default value change the built network at all?** Not *what* it changes - that
is what the per-field tests in `test_unet.py` are for - but that the value arrives. A read
replaced by its own default is exactly the mutation the passes used, and it makes the two
networks identical, which this cannot miss.

Two things make it honest rather than merely mechanical:

* a field whose effect needs a **companion** carries it in the table, because
  `spatial_dropout` does nothing at `dropout: 0.0` and the decoder's `separable_conv` does
  nothing without `upsample: true` - that second one is how the tenth pass's encoder read
  was held while the decoder's identical read was not;
* the table is checked against the fields the builders **actually read**, so a new read
  either gets a row or is named in `NOT_A_STRUCTURAL_READ` with a reason.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

from pyplatypus.models import build_model
from pyplatypus.models.classifier import build_classifier
from pyplatypus.spec.classification import ClassificationModel
from pyplatypus.spec.models import SegmentationModel

MODELS = pathlib.Path(__file__).resolve().parent.parent.parent / "pyplatypus" / "models"

#: field -> (non-default value, companions needed to make it reachable)
SEGMENTATION_FIELDS = {
    "dropout": (0.5, {}),
    "batch_normalization": (False, {}),
    "separable_conv": (True, {"upsample": True}),
    "spatial_dropout": (False, {"dropout": 0.5}),
    "activation": ("gelu", {}),
    "initialiser": ("glorot_uniform", {}),
    "block_width": (3, {}),
    "upsample": (True, {}),
    # A third field needing a companion, and the spec REFUSES it without one:
    # `deep_supervision` has no intermediate outputs to supervise on a plain U-net.
    "deep_supervision": (True, {"architecture": "u_net_plus_plus", "blocks": 3}),
}

CLASSIFICATION_FIELDS = {
    "pooling": ("max", {}),
    "head_dropout": (0.5, {}),
    "activation": ("gelu", {}),
    "batch_normalization": (False, {}),
    "dropout": (0.5, {}),
    "spatial_dropout": (False, {"dropout": 0.5}),
    "separable_conv": (True, {}),
    "block_width": (3, {}),
    "initialiser": ("glorot_uniform", {}),
}

#: Reads this test deliberately does not cover, each with the reason, so the completeness
#: check below is a statement rather than a filter somebody widened until it passed.
#:
#: The staleness half of that check earned itself on its first run: this held `rank`, which
#: the builders do read and which is a derived PROPERTY rather than a field, so naming it
#: here was a claim about something that does not exist.
NOT_A_STRUCTURAL_READ = {
    # Shape-bearing: ignoring one of these fails an output-shape test rather than this one,
    # and `test_unet.py` asserts every architecture preserves its input size.
    "blocks": "shape-bearing; every architecture's shape tests cover it",
    "filters": "shape-bearing",
    "channels": "shape-bearing",
    "input_shape": "shape-bearing",
    "architecture": "chooses the builder; `ARCHITECTURES` is parametrised over it",
    # The backbone route, which needs timm and has its own file.
    "encoder": "chooses the contracting path; `test_pretrained.py`",
    "pretrained": "needs a network to observe; spied in `test_pretrained.py`",
    "normalise": "`#385`, held by `test_pretrained.py`'s encoder-route tests",
}


def _reads_in_the_builders() -> set[str]:
    """Every `spec.<name>` the model builders read, by AST rather than by grep."""
    found: set[str] = set()
    for path in sorted(MODELS.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "spec"
            ):
                found.add(node.attr)
    assert found, "no `spec.<field>` reads found; the scan is looking in the wrong place"
    return found


def _fingerprint(net) -> tuple:
    """What the built network is, closely enough that ignoring a field cannot hide.

    The module types, the parameter names and shapes, and the parameter VALUES - the last
    because `initialiser` changes nothing structural, and a fingerprint that stopped at the
    shapes would report it as delivered while it was being ignored.
    """
    kinds = tuple(sorted(type(module).__name__ for module in net.modules()))
    state = tuple(
        (name, tuple(tensor.shape), round(float(tensor.detach().sum()), 6))
        for name, tensor in sorted(net.state_dict().items())
    )
    return kinds, state


def _segmentation(**overrides):
    base = {"name": "m", "input_shape": (64, 64), "channels": 3, "blocks": 2, "filters": 8}
    torch.manual_seed(0)
    return build_model(SegmentationModel(**{**base, **overrides}), n_class=2)


def _classification(**overrides):
    base = {"name": "c", "input_shape": (64, 64), "channels": 3, "blocks": 2, "filters": 8}
    torch.manual_seed(0)
    return build_classifier(ClassificationModel(**{**base, **overrides}), n_class=3)


@pytest.mark.parametrize("field", sorted(SEGMENTATION_FIELDS))
def test_a_segmentation_field_reaches_the_built_network(field):
    value, companions = SEGMENTATION_FIELDS[field]
    default = _fingerprint(_segmentation(**companions))
    asked = _fingerprint(_segmentation(**companions, **{field: value}))
    assert default != asked, (
        f"`{field}={value!r}` builds the same network as the default, so nothing delivers it"
    )


@pytest.mark.parametrize("field", sorted(CLASSIFICATION_FIELDS))
def test_a_classification_field_reaches_the_built_network(field):
    value, companions = CLASSIFICATION_FIELDS[field]
    default = _fingerprint(_classification(**companions))
    asked = _fingerprint(_classification(**companions, **{field: value}))
    assert default != asked, (
        f"`{field}={value!r}` builds the same network as the default, so nothing delivers it"
    )


def test_the_seed_is_what_makes_the_comparison_mean_anything():
    """The control, and without it every row above passes for the wrong reason.

    `initialiser` is in the table, so the fingerprint reads parameter values - and two
    builds of the SAME specification must then agree, or the comparison is reading the
    draw rather than the field.
    """
    assert _fingerprint(_segmentation()) == _fingerprint(_segmentation())
    assert _fingerprint(_classification()) == _fingerprint(_classification())


def test_every_field_a_builder_READS_is_either_enumerated_or_named():
    """The half that makes this an enumeration rather than a list.

    A field added next month is read by a builder the moment it does anything, and this is
    what fails then. Both passes' survivors were reads nobody enumerated; a test that only
    walks a table cannot notice the row that was never added.
    """
    fields = set(SegmentationModel.model_fields) | set(ClassificationModel.model_fields)
    read = _reads_in_the_builders() & fields
    assert read, "the AST scan found no field reads at all"

    covered = set(SEGMENTATION_FIELDS) | set(CLASSIFICATION_FIELDS) | set(NOT_A_STRUCTURAL_READ)
    missing = sorted(read - covered)
    assert not missing, (
        f"these fields are read by a builder and neither enumerated above nor named in "
        f"NOT_A_STRUCTURAL_READ: {', '.join(missing)}"
    )

    # And the exemptions are real fields rather than a list that drifted.
    stale = sorted(set(NOT_A_STRUCTURAL_READ) - fields)
    assert not stale, f"NOT_A_STRUCTURAL_READ names fields that do not exist: {stale}"
