"""Every optimiser field a builder reads, asked whether it reaches the built optimiser.

`maju116/pyplatypus#404`, and it is the field-delivery class in the one layer no pass had
pointed it at. The tenth and eleventh passes measured model fields: validated by `spec/`,
described in the schema, given `@param` prose in R, and then read **once** by a builder with
nothing asserting that the built object reflects the value. The twelfth pass mutated one read
here - `Adam`'s `weight_decay` replaced by `0.0` - and the **whole suite** passed.

`_BUILDERS` is eight lambdas over 38 `(class, field)` read sites, and not one of them was
held. It is also the easiest route in the package to enumerate, which is the argument for
doing it as a table rather than as a handful of tests: every row is one dictionary entry, and
the completeness half below is read off the lambdas themselves, **attributed to the class
whose key they sit under**. A field added to one optimiser and read in its own builder either
gets a row for that class or fails this file.

The question asked is the weakest one that cannot be satisfied by accident: **does a
non-default value change `optimizer.defaults` at all?** Not which torch key it lands in -
`beta_1` and `beta_2` share `betas`, `learning_rate` is `lr` - because a mapping written here
would be a second copy of the lambda, which is the thing this file is checking.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch
from torch import nn

from pyplatypus.spec.components import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    NAdam,
    RmsProp,
    Sgd,
)
from pyplatypus.training import optimizers as optimizers_module
from pyplatypus.training.optimizers import build_optimizer

#: field -> (a value that is not the default, companions that make it legal)
#:
#: One value per field name rather than per class, because a field means the same thing in
#: every optimiser that has it - and `nesterov` carries a companion for the reason torch
#: gives: Nesterov momentum with no momentum is refused, so the row would be unreachable.
NON_DEFAULT = {
    "learning_rate": (0.05, {}),
    "weight_decay": (0.01, {}),
    "beta_1": (0.5, {}),
    "beta_2": (0.9, {}),
    "eps": (1e-5, {}),
    "amsgrad": (True, {}),
    "momentum": (0.5, {}),
    "nesterov": (True, {"momentum": 0.9}),
    "alpha": (0.5, {}),
    "lr_decay": (0.1, {}),
    "rho": (0.5, {}),
}

#: Where one value per field name is wrong, with the reason - and the control below is what
#: found this entry on its first run: `AdamW`'s `weight_decay` defaults to **0.01** rather
#: than to 0, which is the whole difference between AdamW and Adam, so the shared value was
#: that optimiser's own default and the row would have passed against the mutation.
PER_CLASS = {
    ("AdamW", "weight_decay"): (0.05, {}),
}

#: The eight specifications, by the class `_BUILDERS` keys on.
CLASSES = (Adam, AdamW, Sgd, RmsProp, Adagrad, Adadelta, Adamax, NAdam)


def _asked_for(class_name: str, field: str) -> tuple:
    """The value this row asks for, per class where the shared one does not work."""
    return PER_CLASS.get((class_name, field), NON_DEFAULT[field])


def _reads_in_the_builders() -> set[tuple[str, str]]:
    """Every `(class name, field)` the eight lambdas read, by AST rather than by grep.

    Attributed to the dictionary key, which is what makes this a statement about read
    **sites** rather than about field names: `weight_decay` is read eight times and a row
    for one class says nothing about the other seven.
    """
    source = pathlib.Path(optimizers_module.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_BUILDERS"
        ):
            found = {
                (key.id, child.attr)
                for key, value in zip(node.value.keys, node.value.values, strict=True)
                for child in ast.walk(value)
                if isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "s"
            }
            assert found, "`_BUILDERS` was found and reads no `s.<field>` at all"
            return found
    raise AssertionError("no `_BUILDERS` in `training/optimizers.py`; the scan is misaimed")


def _parameters() -> list[nn.Parameter]:
    torch.manual_seed(0)
    return list(nn.Linear(3, 2).parameters())


def _defaults(spec_class, **overrides) -> dict:
    """What the built optimiser was configured with, which is the object this asks about."""
    spec = spec_class(**overrides)
    return dict(build_optimizer(spec, _parameters()).defaults)


#: `(class, field)` pairs, one test each, derived from the classes rather than written out.
ROWS = sorted(
    (cls.__name__, field) for cls in CLASSES for field in cls.model_fields if field in NON_DEFAULT
)


@pytest.mark.parametrize(("class_name", "field"), ROWS, ids=lambda part: str(part))
def test_an_optimiser_field_reaches_the_built_optimiser(class_name, field):
    spec_class = next(cls for cls in CLASSES if cls.__name__ == class_name)
    value, companions = _asked_for(class_name, field)
    default = _defaults(spec_class, **companions)
    asked = _defaults(spec_class, **companions, **{field: value})
    assert default != asked, (
        f"`{class_name}({field}={value!r})` builds an optimiser configured exactly like the "
        f"default one, so nothing delivers that field"
    )


@pytest.mark.parametrize(("class_name", "field"), ROWS, ids=lambda part: str(part))
def test_the_value_a_row_asks_for_is_not_the_default(class_name, field):
    """The control, and the one that makes the rows above readings rather than tautologies.

    A read replaced by its own default is the mutation this file exists for, so a row whose
    value happened to BE the default would pass against it.
    """
    spec_class = next(cls for cls in CLASSES if cls.__name__ == class_name)
    value, _ = _asked_for(class_name, field)
    assert spec_class.model_fields[field].default != value


def test_two_builds_of_one_specification_agree():
    """The second control: `defaults` must be a reading of the specification and nothing
    else, or every row above is comparing two draws."""
    for cls in CLASSES:
        assert _defaults(cls) == _defaults(cls)


def test_every_read_in_the_builders_has_a_row_for_ITS_CLASS():
    """The completeness half, and it is per site rather than per name.

    `#404`'s own finding: an exemption written for a field was true of one of its reads, so
    two others survived behind it. A row here is a `(class, field)` pair, and a field read by
    a ninth builder tomorrow fails this with that builder's name in the message.
    """
    read = _reads_in_the_builders()
    assert read, "the AST scan found no reads at all"

    covered = set(ROWS)
    missing = sorted(read - covered)
    assert not missing, (
        f"these reads are made by a builder and have no row: "
        f"{', '.join(f'{cls}.{field}' for cls, field in missing)}"
    )

    stale = sorted(covered - read)
    assert not stale, (
        f"these rows describe reads no builder makes: "
        f"{', '.join(f'{cls}.{field}' for cls, field in stale)}"
    )

    # And the per-class exceptions are real rows rather than a list that drifted, which is
    # the half the model enumeration learnt by naming a derived property as a field.
    orphans = sorted(set(PER_CLASS) - covered)
    assert not orphans, (
        f"PER_CLASS names rows that do not exist: "
        f"{', '.join(f'{cls}.{field}' for cls, field in orphans)}"
    )


#: The one field no builder passes to torch, with the reason - `#428`'s other direction for
#: this family. The completeness half above asks *every read has a row for its class*, which
#: a field nobody reads satisfies by being invisible: measured 2026-11-07, `Adam.maximize`
#: planted with a description and no forward passed all five enumerating files in this suite.
NOT_A_PARAMETER = {
    "name": "the discriminator `_BUILDERS` keys on, not a value it hands to torch",
}


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_every_FIELD_of_every_optimiser_is_read_or_named(cls):
    """A field on an optimiser specification either reaches torch or says why not.

    This is the direction that catches the forgotten forward, and it is per class because
    the fields differ: `beta_1` exists on four of the eight and `rho` on one, so a union
    would excuse a field on the class that actually has it.
    """
    read = {field for class_name, field in _reads_in_the_builders() if class_name == cls.__name__}
    assert read, f"the scan found no reads for {cls.__name__}, so this asks nothing"

    missing = sorted(set(cls.model_fields) - read - set(NOT_A_PARAMETER))
    assert not missing, (
        f"{cls.__name__} declares {', '.join(missing)} and its builder reads neither - so a "
        "value a user writes reaches torch nowhere, and nothing else in this suite says so"
    )


def test_the_exemption_is_a_field_every_optimiser_has():
    """The staleness half, which both of the other two lists earned on their first run."""
    for cls in CLASSES:
        stale = sorted(set(NOT_A_PARAMETER) - set(cls.model_fields))
        assert not stale, f"NOT_A_PARAMETER names {stale}, which {cls.__name__} does not have"
