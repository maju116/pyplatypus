"""The README's sections, in the order both packages show them.

Same invariant as the reference groups and for the same reason: the two READMEs cannot be
equal - R has to tell its users what happened to `yolo3()` and this package has no
predecessor with users - but they can show the same sections in the same order, so a reader
arriving from the other repository is not starting again.

Checked here rather than left to care, because four claims in this file had gone stale while
nobody was reading it: the version in the banner was four releases old, it said the R surface
did not exist, it said detection was not on PyPI when it had been since 0.3.0a12, and it gave
a transform count of 97 where the measurement is 87. Documentation rots in the direction of
the work having been done.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

# Also in platypus's tests/testthat/test-readme.R. "About the engine" is R's alone - this
# package *is* the engine - and R's two `###` subsections about coming from 0.1.1 have no
# counterpart, because nothing was released before this.
SECTION_ORDER = [
    "Installing",
    "A worked example",
    "The same thing from a file",
    "What is in it",
    "Boxes instead of masks",
    "A backbone instead of the built-in encoder",
    "Learning it",
    "What is not in it yet",
    "About the engine",
    "About the author",
    "Licence",
]


def sections() -> list[str]:
    """Top-level sections, skipping fenced code - a `## ` inside a block is not a heading."""
    found, fenced = [], False
    for line in README.read_text().splitlines():
        if re.match(r"^\s*```", line):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = re.match(r"^## (.+)$", line)
        if match:
            found.append(match.group(1).strip())
    return found


def test_the_sections_are_a_subsequence_of_the_shared_order():
    got = sections()
    assert len(got) >= 8, f"only {len(got)} sections found - the pattern is wrong, or the file is"

    unknown = [title for title in got if title not in SECTION_ORDER]
    assert not unknown, (
        f"sections not in the shared order: {unknown}. Adding one means adding it to "
        "SECTION_ORDER here and in platypus, or naming it what the other README calls it"
    )

    expected = [title for title in SECTION_ORDER if title in got]
    assert got == expected, (
        f"the README is out of the shared order.\n  here:   {got}\n  shared: {expected}"
    )


def test_the_sections_that_must_be_here_are():
    """The subsequence test permits absence, which is how the groups test let drift back in.

    These eight are the ones both packages have something to say about, so a missing one is
    a section somebody deleted rather than a difference between the languages.
    """
    got = set(sections())
    required = set(SECTION_ORDER) - {"About the engine"}

    assert not (required - got), f"sections missing from the README: {sorted(required - got)}"


def test_the_banner_names_no_version():
    """A version in prose goes stale and this one did, by four releases.

    The banner now points at the changelog instead. Nothing here asserts what the version is -
    `tests/test_version.py` does that against three places that are supposed to agree - this
    asserts that the README does not repeat it.
    """
    banner = README.read_text().split("## ", 1)[0]
    versions = re.findall(r"\b\d+\.\d+\.\d+[ab]?\d*\b", banner)

    allowed = {"0.1.0rc2"}  # the TensorFlow package, named so it can still be installed
    assert not (set(versions) - allowed), (
        f"the banner names a version that will go stale: {sorted(set(versions) - allowed)}"
    )
