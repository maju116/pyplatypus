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
import subprocess
import sys

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


@pytest.fixture(scope="module")
def _generated() -> None:
    """Run the site generator, because two About pages are its output rather than sources.

    `CITATION.qmd` is rendered from `CITATION.cff` at docs-build time and gitignored, so in a
    fresh checkout it does not exist. The first version of the test below asserted every page
    entry was a file in the repository and **passed only because this machine had run the
    generator** - four `test` jobs failed on a clean one, which is the local-versus-CI gap in
    its purest form. What the assertion means is "the build produces this", so the build runs.
    """
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_reference.py")],
        check=True,
        capture_output=True,
    )


def test_the_about_section_is_in_the_agreed_order_and_every_page_exists(_generated):
    """The About section, which the test above does not read.

    Two things are pinned, and the second has already gone wrong in the R package.

    The **order** matches `platypus`'s sidebar, so a reader moving between the two sites
    finds the same things in the same places. It cannot be derived here - this repository
    cannot see the other one - so it is written out in both suites, and `ABOUT_PAGES` in
    `tools/build_reference.py` says where it came from. Citation was the one entry the R
    site had and this one did not, until `CITATION.cff` gave this language's equivalent of
    what altdoc renders there from DESCRIPTION.

    Every **page entry names a file that exists**. Quarto renders `file:` from the project
    directory, so a missing one is a sidebar entry that either fails the build or cannot be
    clicked, depending on where it is named - which is how Licence and Changelog broke on
    the R side before a test said so.
    """
    about = next(
        item
        for item in CONFIG["website"]["sidebar"]["contents"]
        if isinstance(item, dict) and item.get("section") == "About"
    )

    assert [entry["text"] for entry in about["contents"]] == [
        "Changelog",
        "Code of conduct",
        "Licence",
        "Citation",
    ], "run tools/build_reference.py - and keep the order platypus uses"

    for entry in about["contents"]:
        if "file" in entry:
            assert (ROOT / entry["file"]).exists(), (
                f"{entry['text']} points at {entry['file']}, which is not in the repository"
            )
        else:
            # A licence file carries no extension, so Quarto cannot make a page of it and it
            # is linked where it lives instead. Nothing here reaches the network.
            assert entry["href"].startswith("https://"), entry


# The reference groups, in the order both sites show them. Each package carries the subset it
# has: `Setting up` is the reticulate bridge and exists only in R, `Masks and volumes` has no
# public counterpart here - measured again after the drawing functions landed, `__all__` holds
# 32 names and the only two with "mask" in them are `overlay_mask` and `plot_masks`, which
# draw rather than read or unite, and sit with R's in `Looking at the results` - and `Records`
# and `When something is wrong` are this package's, because R has no record reader and no
# documented condition classes. `Looking at the results` was R's alone until 0.8.0a1, when
# the engine learnt to draw, and is in both now.
#
# **Asserting the two lists are equal would be asserting something false**, which the project
# notes did for a while: "the same eight groups" was written down and five of eight titles
# matched. A shared order with each side showing its own subset is the thing that is actually
# true, and it still catches the drift that mattered - `export_weights` and `available_weights`
# sat under `Training` in R while this package had a `Weights` group, so the same two functions
# were filed in the one place a reader of both sites would not look.
#
# Also in platypus's tests/testthat/test-reference-groups.R.
GROUP_ORDER = [
    "Setting up",
    "A specification",
    "What a model is made of",
    "The data on disk",
    "Training",
    "Predicting and scoring",
    "Weights",
    "Records",
    "Looking at the results",
    "Masks and volumes",
    "When something is wrong",
]


def test_the_groups_are_a_subsequence_of_the_shared_order():
    titles = [group["title"] for group in GROUPS]

    unknown = [title for title in titles if title not in GROUP_ORDER]
    assert not unknown, (
        f"group titles not in the shared order: {unknown}. Adding one means adding it to "
        "GROUP_ORDER here and in platypus, or naming it what the other side already calls it"
    )

    expected = [title for title in GROUP_ORDER if title in titles]
    assert titles == expected, (
        f"the groups are out of the shared order.\n  here:   {titles}\n  shared: {expected}"
    )


# Where the names that exist on both sides must be filed. The subsequence test above cannot
# do this: deleting a group leaves a shorter subsequence, which is still a subsequence, so it
# passes - verified by mutation. That is exactly the drift this pins against, because it is
# the drift that happened: `export_weights` and `available_weights` were under `Training` in
# R while they were under `Weights` here.
#
# Only names that exist in both packages are listed, so this is a claim about agreement and
# not a second copy of the grouping. Also in platypus's test-reference-groups.R, with the R
# spellings of the same concepts.
SHARED_PLACEMENT = {
    "available_transforms": "What a model is made of",
    "split_dataset": "The data on disk",
    "export_weights": "Weights",
    "available_weights": "Weights",
}


def test_the_shared_names_are_filed_where_the_other_side_files_them():
    located = {name: group["title"] for group in GROUPS for name in group["contents"]}

    wrong = {
        name: (located.get(name), expected)
        for name, expected in SHARED_PLACEMENT.items()
        if located.get(name) != expected
    }
    assert not wrong, (
        "names that both packages have, filed differently (got, expected): "
        f"{wrong}. platypus's suite pins the same four"
    )
