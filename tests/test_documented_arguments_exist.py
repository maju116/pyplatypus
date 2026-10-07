"""Every argument a docstring documents is an argument the function takes.

Found by quartodoc, which refused to be quiet about it: *"Parameter 'model' does not appear
in the function signature"*. Two of them, both written the same afternoon the `Args:`
sections were - `Engine.evaluate_cases` and `Engine.predict` take `model_name` and I
documented `model`.

Nothing else would have caught it. The docstring is prose to Python, the doctests in those
two methods do not exist because both need a built engine over real data, and the reference
site renders whatever the docstring says - so the published page would have told a reader
to pass an argument that raises `TypeError`. That is worse than no documentation, because
it is confident.

The reverse - an argument with no `Args:` entry - is not asserted. It is a gap rather than
a falsehood, and `**kwargs` passed through to another function is a legitimate reason to
leave one out.
"""

import inspect
import re

import pytest

import pyplatypus
from pyplatypus import DetectionEngine, Engine


def public_callables():
    rows = []
    for name in sorted(pyplatypus.__all__):
        if name == "__version__":
            continue
        obj = getattr(pyplatypus, name)
        if callable(obj):
            rows.append((name, obj))
    for cls in (Engine, DetectionEngine):
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("_"):
                rows.append((f"{cls.__name__}.{name}", method))
    return rows


def documented_arguments(obj):
    """The names under an `Args:` section, in the Google style these docstrings use."""
    doc = inspect.getdoc(obj) or ""
    section = re.search(r"^Args:\n(.*?)(?=\n\S|\Z)", doc, re.DOTALL | re.MULTILINE)
    if not section:
        return []
    return re.findall(r"^\s{4}(\*{0,2}[a-z_][a-z0-9_]*):", section.group(1), re.MULTILINE)


WITH_ARGS = [(label, obj) for label, obj in public_callables() if documented_arguments(obj)]


def test_the_scan_found_the_docstrings():
    """Or this passes by looking at nothing, which is the shape of mistake this project
    has made more than once - including in a guard written to prevent it."""
    assert len(WITH_ARGS) > 25


@pytest.mark.parametrize(
    "label", [label for label, _ in WITH_ARGS], ids=[label for label, _ in WITH_ARGS]
)
def test_every_documented_argument_exists(label):
    obj = next(o for lbl, o in WITH_ARGS if lbl == label)
    try:
        real = set(inspect.signature(obj).parameters)
    except (TypeError, ValueError):  # pragma: no cover - a C-level callable
        pytest.skip(f"{label} has no inspectable signature")

    missing = [
        name
        for name in documented_arguments(obj)
        if name not in real and name.lstrip("*") not in real
    ]
    assert missing == [], (
        f"{label} documents {missing}, which it does not take. Its arguments are "
        f"{sorted(real - {'self'})}."
    )
