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


def test_the_readme_pins_no_transform_count():
    """The 3D paragraph must not say how many transforms take a volume.

    Same shape as the version test above and for a sharper reason: that number is the outcome
    of probing the installed albumentations, and one CI matrix of twelve answered 88 where the
    other eleven answered 87. §3C of the instructions is explicit - a number that is not
    portable between platforms must not be pinned at all, only explained - so the only thing
    to assert is that the README does not state one.

    This file's own docstring cited "a transform count of 97 where the measurement is 87" as
    one of the four stale claims it was written for, and then asserted nothing about it: the
    count sat in the README unguarded through the rewrite that changed it. A test whose
    docstring claims more than its body is the kind that reads as cover and is none.
    """
    text = README.read_text()
    counted = re.findall(r"\b\d+\b(?=\s+(?:of\s+)?(?:its\s+|the\s+)?transforms?\b)", text)
    counted += re.findall(r"\btransforms?\b[^.\n]{0,30}?\b(\d+)\s+(?:take|work|support)", text)

    assert not counted, (
        f"the README states a transform count ({counted}), which is not portable - "
        "explain it instead, as the augmentation paragraph now does"
    )
