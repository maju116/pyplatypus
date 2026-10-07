"""The reference index, which nothing else checks.

Two independent things can go wrong and they fail differently.

**A public symbol in no group** is a page the sidebar cannot reach. quartodoc will not
complain: it generates what the sections ask for and knows nothing about what was left out.
Adding a function and forgetting the index is the ordinary way to break a documentation
site, and the R package's `check_pkgdown()` existed for exactly this - §4v put it *before*
the build, because two seconds beats four minutes.

**A stale `_quarto.yml`** is worse, because the site builds and looks right. The grouping
lives in `tools/reference-groups.yml` and `tools/build_reference.py` turns it into two
things that must agree - quartodoc's `sections`, deciding what gets a page, and the
sidebar, deciding what can be clicked. Either can be a build behind.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest
import yaml

import pyplatypus
from pyplatypus import DetectionEngine, Engine

ROOT = pathlib.Path(__file__).resolve().parent.parent
GROUPS = yaml.safe_load((ROOT / "tools" / "reference-groups.yml").read_text())["groups"]
CONFIG = yaml.safe_load(
    "\n".join(
        line for line in (ROOT / "_quarto.yml").read_text().splitlines() if not line.startswith("#")
    )
)


def public_surface() -> set[str]:
    """What a reader can reach: `__all__`, and the engines' methods under their class."""
    names = {name for name in pyplatypus.__all__ if name != "__version__"}
    for cls in (Engine, DetectionEngine):
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("_"):
                names.add(f"{cls.__name__}.{name}")
    return names


LISTED = [entry for group in GROUPS for entry in group["contents"]]


def test_the_scan_found_a_surface():
    """Or this passes by looking at nothing - the shape of mistake this project has made
    more than once, including in a guard written to prevent it."""
    assert len(public_surface()) > 40
    assert len(GROUPS) >= 6


def test_every_public_symbol_is_in_exactly_one_group():
    surface = public_surface()

    missing = sorted(surface - set(LISTED))
    assert missing == [], (
        f"public but in no group, so unreachable from the sidebar: {missing}. "
        f"Add them to tools/reference-groups.yml and run tools/build_reference.py."
    )

    unknown = sorted(set(LISTED) - surface)
    assert unknown == [], (
        f"in a group but not public, so quartodoc has nothing to document: {unknown}"
    )

    duplicated = sorted({entry for entry in LISTED if LISTED.count(entry) > 1})
    assert duplicated == [], f"in more than one group: {duplicated}"


@pytest.mark.parametrize("target", ["sections", "sidebar"])
def test_the_generated_config_matches_the_grouping(target):
    """`_quarto.yml` is generated. A stale one builds a site that looks right and is wrong."""
    titles = [group["title"] for group in GROUPS]

    if target == "sections":
        got = [section["title"] for section in CONFIG["quartodoc"]["sections"]]
        contents = [entry for s in CONFIG["quartodoc"]["sections"] for entry in s["contents"]]
        expected_contents = LISTED
    else:
        reference = next(
            item
            for item in CONFIG["website"]["sidebar"]["contents"]
            if isinstance(item, dict) and item.get("section") == "Reference"
        )
        nested = [
            item for item in reference["contents"] if isinstance(item, dict) and "section" in item
        ]
        got = [item["section"] for item in nested]
        contents = [entry for item in nested for entry in item["contents"]]
        expected_contents = [f"reference/{entry}.qmd" for entry in LISTED]

    assert got == titles, "run tools/build_reference.py - _quarto.yml is stale"
    assert contents == expected_contents, "run tools/build_reference.py - _quarto.yml is stale"
