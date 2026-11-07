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

**`#404`, the twelfth pass, and both halves above were where it bit.** The reason attached
to an exemption was written for a *field* and was true of one *site*: `blocks` and `filters`
were called shape-bearing, which they are on the U-net path and are not on the encoder route,
where no test used a non-default value - so two reads survived behind a sentence about a
third. The unit is the read site, not the name, which is the eleventh pass's own lesson one
table over.

And the question itself was too weak for a field read in several places. `activation` has a
row here and the classifier's **own encoder** read of it survived anyway: `initialise` reads
the same field two lines further down, so replacing the encoder's read by a literal left the
parameter values different and *something had changed*. So a row now says WHICH reading it
claims - the module tree, or the values in it - and a structural row is asked of the tree
alone, where another read of the same field cannot answer for it. Measured: every field in
both tables but `initialiser` changes `repr(net)`, and `initialiser` changes nothing but the
values, so the split is read off the code rather than chosen.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest
import torch

from pyplatypus.models import build_model
from pyplatypus.models.classifier import build_classifier
from pyplatypus.spec.classification import ClassificationModel
from pyplatypus.spec.models import SegmentationModel

MODELS = pathlib.Path(__file__).resolve().parent.parent.parent / "pyplatypus" / "models"

#: What a row claims the field changes, and the reading it is therefore asked of.
#:
#: `STRUCTURE` is the module tree with its constructor arguments - `repr(net)` - which no
#: other read of the same field can answer for, and that is the whole reason the two are
#: separated. `VALUES` is the numbers in it, for a field that restructures nothing.
STRUCTURE = "structure"
VALUES = "values"

#: field -> (non-default value, companions needed to make it reachable, what it changes)
SEGMENTATION_FIELDS = {
    "dropout": (0.5, {}, STRUCTURE),
    "batch_normalization": (False, {}, STRUCTURE),
    "separable_conv": (True, {"upsample": True}, STRUCTURE),
    "spatial_dropout": (False, {"dropout": 0.5}, STRUCTURE),
    "activation": ("gelu", {}, STRUCTURE),
    "initialiser": ("glorot_uniform", {}, VALUES),
    "block_width": (3, {}, STRUCTURE),
    "upsample": (True, {}, STRUCTURE),
    # A third field needing a companion, and the spec REFUSES it without one:
    # `deep_supervision` has no intermediate outputs to supervise on a plain U-net.
    "deep_supervision": (True, {"architecture": "u_net_plus_plus", "blocks": 3}, STRUCTURE),
    # `#404`: these two were exempt as *shape-bearing*, which is a reason about the U-net's
    # output size and not about either read. They change the tree here, so they are rows.
    "blocks": (4, {}, STRUCTURE),
    "filters": (16, {}, STRUCTURE),
}

CLASSIFICATION_FIELDS = {
    "pooling": ("max", {}, STRUCTURE),
    "head_dropout": (0.5, {}, STRUCTURE),
    "activation": ("gelu", {}, STRUCTURE),
    "batch_normalization": (False, {}, STRUCTURE),
    "dropout": (0.5, {}, STRUCTURE),
    "spatial_dropout": (False, {"dropout": 0.5}, STRUCTURE),
    "separable_conv": (True, {}, STRUCTURE),
    "block_width": (3, {}, STRUCTURE),
    "initialiser": ("glorot_uniform", {}, VALUES),
    # `#404`, and this is the route whose own `_encoder_for` the pass mutated: a classifier's
    # output is `(batch, n_class)` whatever these two are, so NO shape test can see them.
    "blocks": (4, {}, STRUCTURE),
    "filters": (16, {}, STRUCTURE),
}

#: Reads this test deliberately does not cover, each with the reason, so the completeness
#: check below is a statement rather than a filter somebody widened until it passed.
#:
#: The staleness half of that check earned itself on its first run: this held `rank`, which
#: the builders do read and which is a derived PROPERTY rather than a field, so naming it
#: here was a claim about something that does not exist.
#: `#404` took `blocks` and `filters` out of it, because the reason given - *shape-bearing* -
#: described the U-net's output size rather than either read, and two reads survived behind
#: it. What is left names the site that holds the field, so a reason here can be checked.
NOT_A_STRUCTURAL_READ = {
    # Genuinely shape-bearing, and the test that would fail is named rather than implied:
    # every fixture in the suite feeds a 3-channel image, so a builder ignoring `channels`
    # fails on the first convolution rather than on a fingerprint.
    "channels": "the first convolution's width; every fixture's input would stop fitting",
    "input_shape": "the grid itself; `test_unet.py` asserts the output size equals it",
    "architecture": "chooses the builder; `ARCHITECTURES` is parametrised over it",
    # The backbone route, which needs timm and has its own file - and which carries its own
    # enumeration with its own completeness half since `#404`.
    "encoder": "chooses the contracting path; `test_pretrained.py`",
    "pretrained": "needs a network to observe; spied in `test_pretrained.py`",
    "normalise": "`#385`, held by `test_pretrained.py`'s encoder-route tests",
}


#: The three spellings a builder reaches a specification by. `#428`: this scan saw the first
#: only, so `self.spec.blocks` in `models/unet.py` was invisible to it - the field has a row
#: anyway, which is why a scan that could not see it still passed. A spelling the scan cannot
#: see is a read it cannot enumerate, and enumeration is the whole claim.
SPEC_NAMES = {"spec", "model_spec", "s"}


def _is_spec(node: ast.expr) -> bool:
    """`spec`, `model_spec`, or any `<something>.spec` - which covers `self.spec`."""
    if isinstance(node, ast.Name):
        return node.id in SPEC_NAMES
    return isinstance(node, ast.Attribute) and node.attr == "spec"


#: field -> (the module that consumes it, what consumes it there). The OTHER direction's
#: exemption list, and `#428` is why it exists.
#:
#: The completeness half at the bottom of this file asked *every read has a row*. A field
#: with no read at all was invisible to it by construction - which is the forgotten forward,
#: the commonest way a field fails to reach the object it configures and the shape nine of
#: the tenth pass's ten survivors had. Measured 2026-11-07: thirteen fields of these two
#: models were neither enumerated nor named, and three fields planted with no forward passed
#: all five enumerating files in this suite.
#:
#: So the direction below is *every field has a row, an exemption, or a consumer named here*,
#: and the module named is checked rather than taken - a reason that cannot be followed is
#: the decoration §4's newest category is about.
CONSUMED_ELSEWHERE = {
    "name": ("engine.py", "the engine keys everything it reports by it"),
    "fit": ("engine.py", "the engine decides whether to train at all"),
    "weights": ("engine.py", "the engine loads them before any builder runs"),
    "loss": ("engine.py", "the objective is built from it, not the network"),
    "metrics": ("engine.py", "built beside the loss and read by the trainer"),
    "augmentation": ("engine.py", "the loader's pipeline, not the module tree"),
    "batch_size": ("engine.py", "the loader's, and `#404` holds detection's own"),
    "splits": ("data/dataset.py", "the loader tiles with it; `#427` records it in the sidecar"),
    "epochs": ("training/trainer.py", "the loop's length"),
    "callbacks": ("training/trainer.py", "built by `build_callbacks`"),
    "optimizer": ("training/trainer.py", "built by the optimiser table, `#404`"),
    "freeze_encoder": ("training/trainer.py", "the trainer sets requires_grad"),
    "encoder_learning_rate": (
        "training/trainer.py",
        "`parameter_groups`, held at three levels since `#396`",
    ),
}


def reads_the_field(text: str, field: str) -> bool:
    """Does this module consume the field, under any spelling a consumer actually uses?

    Three, and the third is `#428`'s own finding while the list below was being written:
    `DetectionModel.anchors` is consumed as `model.anchors_as_tuples`, a derived property, so
    no `.anchors` read exists anywhere outside the specification layer. A checker that only
    looked for the field's own name would have called that reason false and been wrong.

    The digit in the pattern is this session's other one: a first version of this scan wrote
    `[a-z_]+` and lost `beta_1` and `beta_2`, which the optimiser builders do read. §4's
    *when a count comes back zero, check the instrument*.
    """
    return bool(
        re.search(rf"\b(?:self\.)?(?:spec|model_spec|model|s)\.{field}(?:_[a-z0-9_]+)?\b", text)
    )


def _reads_in_the_builders() -> set[str]:
    """Every field the model builders read, by AST rather than by grep.

    Three spellings and one refusal. `getattr(spec, "x")` is a read with a string in it, so
    it is collected; `getattr(spec, name)` is a read this scan **cannot** resolve, and it
    fails here rather than being passed over - otherwise a builder could reach a field in a
    way that makes the completeness halves below silently smaller.
    """
    found: set[str] = set()
    unresolved: list[str] = []
    for path in sorted(MODELS.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and _is_spec(node.value):
                found.add(node.attr)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and node.args
                and _is_spec(node.args[0])
            ):
                named = node.args[1] if len(node.args) > 1 else None
                if isinstance(named, ast.Constant) and isinstance(named.value, str):
                    found.add(named.value)
                else:
                    unresolved.append(f"{path.name}:{node.lineno}")
    assert not unresolved, (
        "a builder reads a field through a `getattr` this scan cannot resolve: "
        f"{', '.join(unresolved)}. Either name the field, or the enumeration below is a "
        "claim about a set that does not include it"
    )
    assert found, "no field reads found; the scan is looking in the wrong place"
    return found


def _structure(net) -> str:
    """The module tree with each module's own constructor arguments.

    `repr` rather than a set of type names, because `Dropout(p=0.5)` and `Dropout(p=0.1)`
    are the same type and a different network - and because this reading must not be
    satisfiable by a second read of the same field somewhere else, which is exactly what
    the parameter values were doing for `activation`.
    """
    return repr(net)


def _values(net) -> tuple:
    """The parameters themselves, for the one field that restructures nothing."""
    return tuple(
        (name, tuple(tensor.shape), round(float(tensor.detach().sum()), 6))
        for name, tensor in sorted(net.state_dict().items())
    )


def _ask(build, field, value, companions, claims):
    """One row, asked of the reading it claims - and of the other one as a control.

    A `VALUES` row is asked **two-sidedly**: the tree must be identical and the numbers in
    it must differ. The first half is what makes `initialiser` a reading rather than a
    second structural row written loosely, and it would fail if that field ever started
    restructuring anything.
    """
    default = build(**companions)
    asked = build(**companions, **{field: value})
    if claims == STRUCTURE:
        assert _structure(default) != _structure(asked), (
            f"`{field}={value!r}` builds the same module tree as the default, so nothing "
            f"delivers it to the one site this row is about"
        )
        return
    assert _structure(default) == _structure(asked), (
        f"`{field}={value!r}` is enumerated as changing the values alone and it changed "
        f"the module tree; the row is now describing something else"
    )
    assert _values(default) != _values(asked), (
        f"`{field}={value!r}` leaves every parameter where the default put it"
    )


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
    _ask(_segmentation, field, *SEGMENTATION_FIELDS[field])


@pytest.mark.parametrize("field", sorted(CLASSIFICATION_FIELDS))
def test_a_classification_field_reaches_the_built_network(field):
    _ask(_classification, field, *CLASSIFICATION_FIELDS[field])


def test_the_seed_is_what_makes_the_comparison_mean_anything():
    """The control, and without it every row above passes for the wrong reason.

    `initialiser` is in the table, so one reading is the parameter values - and two builds
    of the SAME specification must then agree, or the comparison is reading the draw rather
    than the field. Both readings are controlled, because the structural rows would be
    just as meaningless if `repr` carried an address.
    """
    for build in (_segmentation, _classification):
        assert _structure(build()) == _structure(build())
        assert _values(build()) == _values(build())


def test_both_readings_occur_in_the_tables():
    """A split where every row claims the same thing is one reading with extra words.

    The whole-mask pair's lesson, two tables over: a pairing over a family that answers
    alike proves nothing, so the thing to assert is that both answers are given.
    """
    claimed = {
        claims for _, _, claims in (*SEGMENTATION_FIELDS.values(), *CLASSIFICATION_FIELDS.values())
    }
    assert claimed == {STRUCTURE, VALUES}, f"only {claimed} is claimed by any row"


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


def test_every_field_IS_either_enumerated_or_named_whether_a_builder_reads_it_or_not():
    """The direction the forgotten forward arrives from, and `#428`'s whole point.

    The test above asks *every read has a row*, so a field nothing reads needs neither - and
    a field forwarded nowhere is exactly the defect this file exists to catch. Measured when
    this was written: three planted fields that forwarded nowhere passed all five enumerating
    files in this suite, and the only assertion in the whole engine suite that noticed one of
    them was `test_save_spec.py`'s count of thirty, which says *a field arrived* rather than
    *a field is delivered*.

    So every field is accounted for here in one of three ways, and the third names where it
    goes instead.
    """
    fields = set(SegmentationModel.model_fields) | set(ClassificationModel.model_fields)
    covered = (
        set(SEGMENTATION_FIELDS)
        | set(CLASSIFICATION_FIELDS)
        | set(NOT_A_STRUCTURAL_READ)
        | set(CONSUMED_ELSEWHERE)
    )
    missing = sorted(fields - covered)
    assert not missing, (
        f"these fields have no row, no exemption and no named consumer: {', '.join(missing)}. "
        "A field nothing forwards is the shape this file was written for, so it is named here "
        "with where it goes or it gets a row"
    )

    stale = sorted(set(CONSUMED_ELSEWHERE) - fields)
    assert not stale, f"CONSUMED_ELSEWHERE names fields that do not exist: {stale}"


@pytest.mark.parametrize("field", sorted(CONSUMED_ELSEWHERE))
def test_the_consumer_named_for_a_field_really_reads_it(field):
    """A reason that cannot be followed is decoration, which is §4's newest category.

    Both of this file's exemption lists caught their own author on their first run - one
    named a derived property, the other four arguments no constructor takes - so the reason
    beside a field is checked here rather than read. The module is named and not the line,
    because a line number names the place somebody looked rather than the place the code is.
    """
    module, why = CONSUMED_ELSEWHERE[field]
    path = MODELS.parent / module
    assert path.exists(), f"{field} names {module}, which is not a file"
    assert reads_the_field(path.read_text(), field), (
        f"{field} is exempt because {why}, in {module} - and {module} does not read it. "
        "Either the consumer moved or the reason was never true"
    )
