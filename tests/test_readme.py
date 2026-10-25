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
    for line in README.read_text(encoding="utf-8").splitlines():
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
    banner = README.read_text(encoding="utf-8").split("## ", 1)[0]
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
    text = README.read_text(encoding="utf-8")
    counted = re.findall(r"\b\d+\b(?=\s+(?:of\s+)?(?:its\s+|the\s+)?transforms?\b)", text)
    counted += re.findall(r"\btransforms?\b[^.\n]{0,30}?\b(\d+)\s+(?:take|work|support)", text)

    assert not counted, (
        f"the README states a transform count ({counted}), which is not portable - "
        "explain it instead, as the augmentation paragraph now does"
    )


def test_the_readme_pins_no_count_of_example_programs():
    """And does not enumerate the R package's vignettes, for the same reason.

    Both were wrong at once, found on 2026-10-15 while fixing the R README's "Three
    vignettes" that named four. This page said `examples/` holds "the four programs the
    measurements on this page came from" - the instance paragraph's 0.7655, +0.068 and
    +0.2398 come from three more - and that the R vignettes are "blood cells, nuclei,
    volumes", which has omitted the retinal one since it landed.

    Neither number nor list is holdable from here: the vignettes live in the other
    repository, and which scripts a table came from is a judgement about this page's prose.
    §3C's rule applies to both - a number in prose needs a test holding it to the
    measurement, or no number at all - so what is asserted is the absence.
    """
    text = README.read_text(encoding="utf-8")

    counted = re.findall(r"\b(?:two|three|four|five|six|\d+)\s+programs?\b", text, re.IGNORECASE)
    assert not counted, (
        f"the README counts example programs ({counted}); name each one beside its table "
        "instead, which is what makes a table reproducible"
    )

    listed = re.findall(r"vignettes\]\([^)]*\)[^.]{0,80}?-\s*[a-z ]+,", text)
    assert not listed, (
        f"the README enumerates the R vignettes ({listed}), which nothing here can hold: "
        "they are in the other repository and a new one leaves this list stale"
    )


def test_every_example_file_the_readme_names_exists():
    """A dangling script name is the cheap half of the same defect, and this is holdable.

    The sentence above promises every table is reproducible by running the script named
    beside it, so the names have to resolve. Controlled by renaming one in a scratch copy:
    the assertion fires.
    """
    text = README.read_text(encoding="utf-8")
    named = sorted(set(re.findall(r"`(examples/[A-Za-z0-9_./-]+)`", text)))
    assert len(named) > 3, f"the scan found {len(named)} example paths, so it found nothing"

    missing = [name for name in named if not (ROOT / name).exists()]
    assert not missing, f"the README names example files that do not exist: {missing}"


#: The number words the feature list is written in, because "Eleven losses" reads better in
#: a bullet than "11 losses" and a test should not force the prose to change register.
NUMBERS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
}


def _menu_name(written: str) -> str:
    """A name as the feature list writes it, as the menu spells it.

    `CCE-Dice` is `cce_dice`, `clDice` is `cldice`, `Lovász` is `lovasz`. Normalising here
    rather than demanding the README use snake_case keeps the page readable and still makes
    the comparison exact.

    The accent is stripped by decomposing and dropping the combining marks rather than by
    replacing the one character this list happens to contain. The literal version passed
    here and **failed on both Windows jobs**: `Path.read_text()` with no encoding reads the
    file in the platform's, so on Windows the UTF-8 bytes of `á` arrived as something else
    and the replacement matched nothing. Both halves of that are fixed - every read in this
    file now names UTF-8 - and the general form is kept because the next accented name
    would otherwise be the same bug again.
    """
    import re as _re
    import unicodedata

    # Whitespace collapsed first: the bullet wraps, so a two-word name arrives with the
    # newline and the next line's indent in the middle of it.
    squeezed = _re.sub(r"\s+", " ", written).strip().lower()
    decomposed = unicodedata.normalize("NFKD", squeezed)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return plain.replace("-", "_").replace(" ", "_")


def test_the_feature_list_counts_the_losses_and_metrics_the_engine_offers():
    """Both counts and both lists of names, against the menus themselves.

    This is the one count in the README that *is* holdable, which is why it gets a pair
    rather than the absence the two tests above assert. The transform count depends on the
    installed albumentations and the vignettes live in the other repository; `loss` and
    `metrics` are this package's own menus, so `available_losses()` can simply be asked.

    It was not asked, and the bullet rotted the way §4 says documentation rots - in the
    direction of the work having been done. It read "Nine losses ... and three metrics"
    while the engine offered eleven and five: `boundary` and `generalized_dice` had arrived
    among the losses, `surface_dice` and `tversky` among the metrics, and the first four
    claims this file was written for were the same mistake in the same file.

    Both halves are asserted, because either alone passes for the wrong reason: a count
    with no names goes stale silently the moment a loss is renamed, and names with no count
    pass while the prose says nine of them.
    """
    import re as _re

    from pyplatypus import available_losses, available_metrics

    text = README.read_text(encoding="utf-8")

    for noun, menu in (("losses", available_losses()), ("metrics", available_metrics())):
        found = _re.search(rf"\*\*(\w+) {noun}\*\*\s*\(([^)]*)\)", text.replace("\n", " "))
        assert found is not None, (
            f"the feature list no longer states the {noun} as `**N {noun}** (a, b, c)`; "
            "this test reads that shape, so either restore it or rewrite the test with it"
        )
        written = found.group(1).lower()
        assert written in NUMBERS, f"{written!r} is not a number word this test knows"
        offered = {row["name"] for row in menu}
        assert NUMBERS[written] == len(offered), (
            f"the README says {written} {noun} and the engine offers {len(offered)}: "
            f"{sorted(offered)}"
        )
        named = {_menu_name(part) for part in found.group(2).split(",")}
        assert named == offered, (
            f"the {noun} the README names are not the {noun} the engine offers.\n"
            f"  only in the README: {sorted(named - offered)}\n"
            f"  only in the engine: {sorted(offered - named)}"
        )
