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

**And a docstring can carry the same section twice**, which is `#312`: `crops()` had two
`Args:` blocks and two `Returns:` blocks, and neither pair agreed with itself - one `Returns:`
omitted `dropped` and the other omitted `scores`, `labels` and `names`, so whichever a reader
believed, they were wrong about the function's answer. Nothing in this repository read a
docstring's *shape*: `quartodoc build` renders it without complaining and the `docs` job was
green the whole time.

It also blinded the instrument above. `documented_arguments` reads the **first** `Args:` block
and stops, so a second block's entries are never checked against the signature - a parameter
invented there would have been invisible to the one test written to catch exactly that.
"""

import ast
import inspect
import re
from collections import Counter
from pathlib import Path

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


def documented_default_of_the_first():
    """Every engine method whose `Args:` says `model_name` defaults to the first."""
    rows = []
    for engine in (Engine, DetectionEngine):
        for name, function in inspect.getmembers(engine, inspect.isfunction):
            if name.startswith("_"):
                continue
            signature = inspect.signature(function)
            if "model_name" not in signature.parameters:
                continue
            claims = "Defaults to the first" in (function.__doc__ or "")
            rows.append((f"{engine.__name__}.{name}", claims, signature))
    return rows


@pytest.mark.parametrize(
    ("where", "claims", "signature"),
    documented_default_of_the_first(),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_documented_default_is_a_default(where, claims, signature):
    """Nine pages said `model_name` "defaults to the first" and nine signatures required it.

    The same mistake as the one above and a wider one: not an argument that does not exist,
    but an argument whose documented *default* did not. Omitting the name raised `TypeError`
    on all nine, and `evaluate_cases`, `predict`, `report`, `crops`, `box_shapes`,
    `anchor_coverage`, `evaluate_classes` and `evaluate_images` all carried the claim.

    Asserted both ways. `export_weights` documents no default and must not grow one by
    accident either, because its second parameter is required and positional - a default on
    the first would be a syntax error, which is why those two were the only honest pair.
    """
    default = signature.parameters["model_name"].default
    if claims:
        assert default is None, f"{where} documents a default and has none"
    else:
        assert default is inspect.Parameter.empty, (
            f"{where} takes a default for `model_name` and does not document it"
        )


# --------------------------------------------------------------- one section, once

# The Google-style headers griffe recognises, plus the numpydoc spellings, because a
# docstring written in the wrong dialect is the way a second block gets added without
# anybody seeing a duplicate. A header is a line that is *only* the word and a colon -
# `Args: the ones below` is prose and is left alone.
SECTION_HEADER = re.compile(
    r"^[ \t]*(Args|Arguments|Attributes|Example|Examples|Methods|Note|Notes|Other Parameters"
    r"|Parameters|Raises|Receives|References|Returns|See Also|Todo|Warning|Warnings|Warns"
    r"|Yields):[ \t]*$",
    re.MULTILINE,
)


def repeated_sections(doc: str) -> dict[str, int]:
    """Which section headers this docstring carries more than once, and how often."""
    counts = Counter(match.group(1) for match in SECTION_HEADER.finditer(doc))
    return {name: n for name, n in sorted(counts.items()) if n > 1}


def docstrings_in_the_package():
    """Every docstring in `pyplatypus/`, by AST rather than by import.

    Wider than `public_callables()` above on purpose, and the width is the point: that scan
    reads `__all__` plus two engines, so `ClassificationEngine` and `InstanceEngine` are
    outside it, and a module docstring is outside it however it is written. The published
    reference renders all of them.
    """
    root = Path(pyplatypus.__file__).parent
    rows = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(
                node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                continue
            # `clean=False`: the indentation is part of what a section header is, and
            # `inspect.getdoc`'s dedent would make a nested block look like a top-level one.
            doc = ast.get_docstring(node, clean=False)
            if doc:
                label = f"{path.relative_to(root.parent)}:{getattr(node, 'lineno', 1)} "
                rows.append((label + getattr(node, "name", "<module>"), doc))
    return rows


DOCSTRINGS = docstrings_in_the_package()


def test_the_section_scan_found_the_docstrings():
    """The floor is deliberately far below the count rather than near it.

    What it guards is a scan that walked nothing and reported no duplicates, which is this
    project's commonest shape of false green. **No exact number**, because this one moves
    with every commit that adds a function - the first draft of this docstring pinned one
    and was stale inside the hour, by the helper added in the same pair of changes.
    """
    assert len(DOCSTRINGS) > 300


def test_the_detector_fires_on_a_duplicate_and_not_on_prose():
    """The positive control, and the negative one beside it.

    Without the first, every assertion below passes on a regex that matches nothing. Without
    the second, the check is tight enough to fail on correct content, which §4 calls as
    useless as loose enough to pass on wrong content.
    """
    twice = "Does a thing.\n\n    Args:\n        a: one\n\n    Args:\n        b: two\n    "
    assert repeated_sections(twice) == {"Args": 2}

    once = "Does a thing.\n\n    Args:\n        a: one\n\n    Returns:\n        Something.\n    "
    assert repeated_sections(once) == {}

    prose = "Args: the four below are the only ones.\n\n    Args: and these are not.\n    "
    assert repeated_sections(prose) == {}


def test_no_docstring_carries_the_same_section_twice():
    offenders = [
        f"{label} {repeated}" for label, doc in DOCSTRINGS if (repeated := repeated_sections(doc))
    ]
    assert offenders == [], (
        "a docstring carrying one section twice publishes two answers to one question, and "
        "the reference renders both:\n  " + "\n  ".join(offenders)
    )
