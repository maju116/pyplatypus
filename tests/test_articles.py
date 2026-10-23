"""An engine article walks the same sections as the R vignette it mirrors.

`maju116/rplatypus#117` settles the direction: **the vignette is the source and the article
is the copy**. So this test lives on the engine's side and fetches R's file, which is the
mirror image of R's `test-brand.R` fetching the engine's `_brand.yml` - the copy side asks
the source side, in both cases, and neither repository reaches into the other's working tree.

What it does NOT assert, and this is the design rather than an omission. The two pages share
a walk, not a body: measured on the only pair that exists, `guide.qmd` and
`configuration.Rmd` carry the same ten `##` headings in the same order, the same YAML blocks
verbatim because YAML belongs to neither language, and **2 blocks of Python against 11 of R**.
`maju116/pyplatypus#237` asks for exactly that difference in its own words - the engine
reaches a destination through a configuration file where R reaches it through constructors,
and a straight translation would waste that. So the headings are pinned and the route is not.

`###` headings are deliberately not pinned either, though the one existing pair happens to
agree on its single sub-heading. Sub-structure is where the route differs, and a constraint
written now for five unwritten articles would be a guess about documents nobody has read.

The cases come from the files rather than from a list. Every `*.qmd` at the repository root
whose front matter carries `mirrors:` is checked, so **a new article is covered by arriving**
- which is CLAUDE.md §1's lesson from `tau` and `surface_dice`, both of which were passing
tests whose fixture never contained the new case because the cases were listed by hand.

This does not replace `test_guide.py`'s `SECTIONS`. That list is the same claim without a
network, and the two fail for different reasons: the literal catches the engine's page
changing, this catches the two drifting apart. A suite with no network keeps the first.
"""

from __future__ import annotations

import pathlib
import re
import urllib.error
import urllib.request

import pytest

from tests.conftest import NOT_OURS, every_name_this_package_has

ROOT = pathlib.Path(__file__).resolve().parent.parent

RAW = "https://raw.githubusercontent.com/maju116/rplatypus/main"
VIGNETTE = RAW + "/vignettes/{name}.Rmd"
# The control for a failed fetch, exactly as `test-brand.R` uses the engine's README: if this
# is reachable and a vignette is not, the vignette moved and that is a finding. Only both
# unreachable is a skip, because a network probe that answers "no" for its own reasons is
# §4's category in its commonest form.
CONTROL = RAW + "/README.md"


def articles() -> list[tuple[pathlib.Path, str]]:
    """Every root `.qmd` that declares the vignette it mirrors, with that vignette's name."""
    found = []
    for path in sorted(ROOT.glob("*.qmd")):
        head = path.read_text().split("---", 2)
        if len(head) < 3:
            continue
        declared = re.search(r"^mirrors:\s*(\S+)\s*$", head[1], re.MULTILINE)
        if declared:
            found.append((path, declared.group(1)))
    return found


def fetch(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def sections(text: str) -> list[str]:
    return re.findall(r"^## (.+)$", text, re.MULTILINE)


ARTICLES = articles()


def test_the_scan_found_articles_at_all():
    """Or every assertion below passes by looking at nothing.

    The guide declares `mirrors: configuration`, so this is >= 1 today and grows with the
    milestone. A zero here means the front-matter pattern is wrong, not that no article
    mirrors anything - and that failure would be silent in every other test in this file.
    """
    assert ARTICLES, (
        "no .qmd at the repository root declares `mirrors:` in its front matter - "
        "the pattern is wrong, or an article lost its declaration"
    )


@pytest.mark.parametrize(("path", "vignette"), ARTICLES, ids=[path.name for path, _ in ARTICLES])
def test_an_article_walks_its_vignettes_sections(path, vignette):
    theirs = fetch(VIGNETTE.format(name=vignette))
    if theirs is None:
        if fetch(CONTROL) is not None:
            pytest.fail(
                f"{CONTROL} is reachable and vignettes/{vignette}.Rmd is not. "
                f"{path.name} says it mirrors `{vignette}`, so either the vignette was "
                "renamed or the declaration is wrong - this is not a network problem"
            )
        pytest.skip(f"no network: vignettes/{vignette}.Rmd could not be fetched")

    ours, source = sections(path.read_text()), sections(theirs)
    assert source, (
        f"vignettes/{vignette}.Rmd was fetched and has no `##` headings at all, "
        "which is the extractor failing rather than the vignette being empty"
    )
    assert ours == source, (
        f"{path.name} and vignettes/{vignette}.Rmd have come to walk different paths.\n"
        f"  only here:  {[s for s in ours if s not in source]}\n"
        f"  only there: {[s for s in source if s not in ours]}\n"
        "The vignette is the source (rplatypus#117), so the article follows it - "
        "or both change together and deliberately."
    )


MENTIONED = sorted(
    {
        name
        for path, _ in ARTICLES
        for name in re.findall(r"`([a-z_][a-z0-9_]*)\(\)`", path.read_text())
        if name not in NOT_OURS
    }
)


def test_the_articles_name_functions_at_all():
    """The same guard as above, for the other extractor."""
    assert len(MENTIONED) > 3, (
        f"found {len(MENTIONED)} function names across {len(ARTICLES)} article(s) - "
        "the backtick pattern is wrong"
    )


@pytest.mark.parametrize("name", MENTIONED)
def test_a_name_an_article_offers_a_reader_resolves(name):
    """§4's "an instruction addressed to a human is code nothing runs", pointed at prose.

    An article's code blocks are static text held by tests rather than executed, which is
    what `pyplatypus#172` settled and what the guide already does. That buys a page that
    renders without torch and costs the one thing execution would have given for free: a
    name that stopped existing is just prose. This is that cost paid back.
    """
    assert name in every_name_this_package_has(), (
        f"an article names `{name}()` and this package has no such name. The reader meets it "
        "as something to call. If it is the R package's, say what the engine answers with "
        "instead - these pages are read by people who have no R."
    )
