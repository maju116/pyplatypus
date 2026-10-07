"""The package's own front door.

Two things nothing else checks. `__all__` is what `from pyplatypus import *` offers and
what a reader takes for the public surface, so an entry that does not resolve is a
promise the package cannot keep. And a message telling somebody to call
`available_transforms(rank=3)` is worth nothing if that name is unreachable from the
package root, which it was until 0.7.0a1 - the function lived in
`pyplatypus.spec.components` and the message never said so.
"""

import re
from pathlib import Path

import pytest

import pyplatypus


def test_every_name_in_all_resolves():
    missing = [name for name in pyplatypus.__all__ if not hasattr(pyplatypus, name)]
    assert missing == []


def test_all_is_sorted_so_an_addition_lands_in_one_place():
    assert list(pyplatypus.__all__) == sorted(pyplatypus.__all__)


@pytest.mark.parametrize("name", ["available_transforms", "available_weights"])
def test_the_listings_are_reachable_from_the_root(name):
    """`available_*` is one verb for everything a user can ask to be shown. The R package
    exports the same two names, so a reader moving between the halves finds them."""
    listing = getattr(pyplatypus, name)()
    assert len(listing) > 0


def test_every_function_an_error_message_names_is_reachable():
    """The rule this file exists for, applied to the whole package rather than to the one
    message that broke it: any `name(...)` a user-facing string tells somebody to call has
    to be importable from `pyplatypus`, or the message sends them hunting for a module
    path it does not give.

    Only names the package itself defines are considered - `len(x)` in a message is not a
    promise about `pyplatypus.len`. Private names are not either: nobody is told to call
    `_nearest`.

    An f-string's `{...}` spans are removed first. They are code, not text, and leaving
    them in reported three calls made *inside* messages as though they had been quoted to
    the reader - the test finding its own author out before it found anything else.
    """
    root = Path(pyplatypus.__file__).parent
    own = set()
    for path in root.rglob("*.py"):
        own.update(re.findall(r"^def ([a-z][a-z0-9_]*)\(", path.read_text(), re.MULTILINE))

    unreachable = {}
    for path in root.rglob("*.py"):
        for message in re.findall(r'"([^"\\\n]{20,})"', path.read_text()):
            prose = re.sub(r"\{[^{}]*\}", "", message)
            for called in re.findall(r"\b([a-z][a-z0-9_]*)\(", prose):
                if called in own and not hasattr(pyplatypus, called):
                    unreachable.setdefault(called, set()).add(path.name)

    assert unreachable == {}, (
        "a message names these and they cannot be imported from pyplatypus: "
        + ", ".join(f"{n} ({', '.join(sorted(f))})" for n, f in sorted(unreachable.items()))
    )
