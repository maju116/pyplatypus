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
import yaml

from pyplatypus import ConfigError, from_dict
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


# --------------------------------------------------------------------------------------
# The article's code blocks, which nothing checked until one of them was wrong.
#
# `instances.qmd` shipped with `build_engine(...)` followed straight by
# `predict_instances(...)`, and the engine refuses that: with `fit: false` the weights are
# loaded by `fit()`, so `predict` before it answers *no model has been trained or loaded
# yet*. A reader copying the block would have been taught a mistake, which is exactly what
# `test_the_guide_shows_configurations_at_all`'s sibling was written for on the YAML side -
# and the gap was named in that pull request's own body an hour before it was fallen into.
#
# These two do not execute anything. Executing an article means downloading weights and
# touching torch, which is not what a suite should do; what is cheap is asking whether the
# narrative obeys the engine's own lifecycle, and whether a configuration it shows would be
# accepted.

CALL = re.compile(r"\bengine\.(\w+)\(")


def python_blocks(path: pathlib.Path) -> list[tuple[int, str]]:
    text = path.read_text()
    found = []
    for match in re.finditer(r"^```python\n(.*?)^```", text, re.MULTILINE | re.DOTALL):
        found.append((text[: match.start()].count("\n") + 1, match.group(1)))
    return found


BLOCKS = [(path, line, block) for path, _ in ARTICLES for line, block in python_blocks(path)]

# `fit` is what loads or trains; everything that reads a model needs it to have happened.
NEEDS_A_MODEL = {"predict", "predict_instances", "evaluate", "evaluate_cases"}


@pytest.mark.parametrize(("path", "vignette"), ARTICLES, ids=[p.name for p, _ in ARTICLES])
def test_an_article_shows_python_at_all(path, vignette):
    """The guard, because an extractor that matches nothing passes both tests below.

    **Per article, not over the total, and that correction is the point.** The first
    version asserted `len(BLOCKS) >= 2` across every page, and `guide.qmd` alone has
    exactly two - so emptying the other article's blocks left the threshold satisfied by a
    different file and the guard said nothing. Measured: turning every ```python fence in
    `instances.qmd` into ```text gave fifteen passes and no failure. §4's "a check
    calibrated loosely enough admits the failure it was written to prevent", inside the
    guard written against that failure.

    An article narrates a run, so one block is the floor for all of them; a page with none
    is a page that stopped doing what `pyplatypus#172` settled these are for.
    """
    blocks = python_blocks(path)
    assert blocks, (
        f"{path.name} shows no ```python blocks - the fence pattern is wrong, or the page "
        "stopped narrating a run"
    )


@pytest.mark.parametrize(
    ("path", "line", "block"),
    BLOCKS,
    ids=[f"{path.stem}-line-{line}" for path, line, _ in BLOCKS],
)
def test_a_block_that_reads_a_model_has_fitted_first(path, line, block):
    """Within one block: if it builds an engine and reads a model, `fit` is between them.

    A block that only predicts is narrating a continuation of an earlier one and is left
    alone - the alternative is demanding `fit()` in every snippet, which would make the
    pages read like a transcript.
    """
    calls = CALL.findall(block)
    if "build_engine" not in block:
        return
    reads = [call for call in calls if call in NEEDS_A_MODEL]
    if not reads:
        return
    assert "fit" in calls, (
        f"{path.name} line {line} builds an engine and then calls "
        f"{reads[0]}() with no fit() between them. With `fit: false` it is fit() that "
        "loads the weights, so the engine answers 'no model has been trained or loaded "
        "yet' - a reader copying this block is taught a mistake"
    )


CONFIGS = [
    (path, line, found.group(1))
    for path, line, block in BLOCKS
    for found in [re.search(r"from_dict\(\s*(\{.*\})\s*\)", block, re.DOTALL)]
    if found
]


@pytest.mark.parametrize(
    ("path", "line", "literal"),
    CONFIGS,
    ids=[f"{path.stem}-line-{line}" for path, line, _ in CONFIGS],
)
def test_a_configuration_an_article_shows_is_accepted(path, line, literal):
    """The guide's YAML check, pointed at the dicts the articles hand to `from_dict`.

    `check_paths=False`: the paths in an article are illustrative, the fields are not.
    `ast.literal_eval` rather than `eval`, because a page is text and text is not trusted
    just because it is in this repository.
    """
    import ast

    try:
        config = ast.literal_eval(literal)
    except (ValueError, SyntaxError) as broken:
        pytest.fail(f"{path.name} line {line} shows a dict that is not a literal: {broken}")
    try:
        from_dict(config, check_paths=False)
    except ConfigError as refusal:
        pytest.fail(f"the configuration at {path.name} line {line} is refused:\n{refusal}")


# --------------------------------------------------------------------------------------
# The other fence. `guide.qmd` has had its YAML validated since it was written; an article
# that shows a configuration file had nothing, and a configuration is exactly what an engine
# article reaches a destination *through* - `pyplatypus#237`'s whole distinction from R's
# constructors. So the same check, pointed at the same kind of block wherever it appears.
#
# This is `#245`'s lesson applied one fence over: a block a reader copies is a claim, and the
# page had no way of being wrong out loud.


def yaml_configurations(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every fenced yaml block in one article that is a configuration, with its line.

    A `yaml` fence with no `task:` is something else - a fragment, an editor's schema
    comment - and validating it would fail for a reason the page is not claiming anything
    about. The same rule `test_guide.py` uses, so the two agree about what a configuration
    block is.
    """
    text = path.read_text()
    found = []
    for match in re.finditer(r"^```yaml\n(.*?)^```", text, re.MULTILINE | re.DOTALL):
        if re.search(r"^task:", match.group(1), re.MULTILINE):
            found.append((text[: match.start()].count("\n") + 1, match.group(1)))
    return found


YAML_BLOCKS = [
    (path, line, block) for path, _ in ARTICLES for line, block in yaml_configurations(path)
]


WITH_YAML = [(path, vignette) for path, vignette in ARTICLES if "```yaml" in path.read_text()]


@pytest.mark.parametrize(("path", "vignette"), WITH_YAML, ids=[path.name for path, _ in WITH_YAML])
def test_the_yaml_extractor_sees_every_fence_in_an_article(path, vignette):
    """The guard, and the first version of it asserted something false.

    It demanded a configuration from every article carrying a `yaml` fence, and
    `data-science-bowl.qmd` carries four **fragments** - a `models:` list, a `callbacks:`
    block - which have no `task:` and are not configurations. The page was right and the
    guard was wrong, which is the cheap direction to find out in.

    What is actually fragile here is the fence pattern, so that is what is checked, against
    a count nothing clever can get wrong. The `task:` filter is checked by the floor below
    instead: it is simple, and its failure mode is finding nothing anywhere.
    """
    text = path.read_text()
    fences = len(re.findall(r"^```yaml\n(.*?)^```", text, re.MULTILINE | re.DOTALL))
    assert fences == text.count("```yaml\n"), (
        f"{path.name} has {text.count('```yaml')} yaml fences and the pattern matched "
        f"{fences} - an unterminated fence, or the pattern is wrong"
    )


def test_an_article_somewhere_shows_a_whole_configuration():
    """The other half: that `task:` still selects something.

    A floor rather than a per-article demand, because showing a whole configuration is one
    route of several - `instances.qmd` hands `from_yaml()` a file in the repository and
    shows no YAML at all, which is a legitimate way for an article to reach the same place.
    """
    assert YAML_BLOCKS, (
        "no article shows a complete configuration - either `task:` stopped matching, or "
        "every page that had one lost it"
    )


@pytest.mark.parametrize(
    ("path", "line", "block"),
    YAML_BLOCKS,
    ids=[f"{path.stem}-line-{line}" for path, line, _ in YAML_BLOCKS],
)
def test_a_yaml_configuration_an_article_shows_is_accepted(path, line, block):
    """`check_paths=False`: the paths in an article are illustrative, the fields are not."""
    try:
        parsed = yaml.safe_load(block)
    except yaml.YAMLError as broken:
        pytest.fail(f"{path.name} line {line} shows yaml that does not parse: {broken}")
    try:
        from_dict(parsed, check_paths=False)
    except ConfigError as refusal:
        pytest.fail(f"the configuration at {path.name} line {line} is refused:\n{refusal}")


# --------------------------------------------------------------------------------------
# The figures the two retinal pages quote OF EACH OTHER - `pyplatypus#415`.
#
# This is a §3F pair nobody wrote down. `retinal-vessels.qmd` reports a forty-epoch run and
# argues from R's fifteen-epoch one; `vignettes/retinal-vessels.Rmd` reports the fifteen and
# argues from the forty, **quoting two of its figures to three decimals**. Each page's claim
# rests on a number printed on the other, both pages are right to argue that way - *the
# better-trained model closes the gap* is the finding and it needs both runs - and until this
# test nothing noticed when one of the two moved.
#
# It moved. The vignette was re-knitted for `maju116/rplatypus#289` under `0.8.0a73` and its
# mean difference went from -0.0523 to -0.0833, so the article's *about five points below*
# became about eight and said five until somebody read both pages. Neither instrument nearby
# can see it: R's `test-precomputed-vignettes.R` compares a vignette against its own source,
# and the prose test above resolves `name()` and reads no figures at all.
#
# WHAT IS HELD, since the issue's own question was *which figures*. The two pages do not print
# the same quantity and should not - one ran forty epochs and the other fifteen:
#
#   the sign        each page's own gap, in the direction its own prose claims
#   the ordering    the forty-epoch gap sits above the fifteen-epoch one, which IS the
#                   shared claim and the only thing the two pages assert together
#   a quoted NUMBER held to the other page's own output, wherever a page states one
#   a quoted WORD   held in its direction alone, because *several points below* is a claim
#                   about a sign and about nothing else
#
# That last distinction is the whole design: it survives a re-knit, which the issue asks for,
# **and** still fails on the stale *five*, which a sign-only guard would have passed. A number
# in prose is pinned; a direction is not.
#
# This pair is NAMED rather than scanned, unlike everything above it in this file. There is no
# general rule that an article quotes its vignette's figures - this one pair does, because its
# two runs differ in epochs on purpose. A second such pair earns its own §3F row and its own
# extractor, and a scan written now for pages nobody has written would be a guess.

RETINAL = "retinal-vessels"
RETINAL_ARTICLE = ROOT / "retinal-vessels.qmd"

# The prose writes a count of points as a word, which is the house style in both packages.
# `several`, `a few` and `some` are deliberately absent rather than mapped to a number: they
# are not numbers, and the word/number distinction above only means something if a page that
# names no number pins nothing.
NUMBER_WORDS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
}

# A point is a hundredth, so a mean difference of -0.0833 is 8.33 points. The bound is one
# point: twice the most an honest rounding can be out - half a point - and a third of the
# defect this exists for, *five* against 8.33, which is 3.33 away. Both figures are measured
# off the pages as they stood when `#415` was filed, which is what §4 asks of a tolerance:
# above the noise and below the thing it is written to catch, with both numbers written down.
POINT = 1.0


def r_printed_figures(text: str) -> dict[str, float]:
    """The figures the vignette's own chunk prints, read by the names printed above them.

    By role rather than by position. knitr writes a named vector as two `#>` lines, the first
    carrying the names, so a reordered `round(c(...))` still answers correctly and a renamed
    one fails loudly rather than reading a neighbour's column.
    """
    printed = [line[2:] for line in text.splitlines() if line.startswith("#>")]
    for index, header in enumerate(printed[:-1]):
        names = header.split()
        if {"correlation", "mean_difference"} <= set(names):
            values = printed[index + 1].split()
            if len(values) == len(names):
                return dict(zip(names, (float(value) for value in values)))
    return {}


def article_table_figures(text: str) -> dict[str, float]:
    """The article's own two figures, read out of its table by the row labels."""
    labels = {
        "correlation": "correlation of dice and cldice",
        "mean_difference": "mean of clDice minus Dice",
    }
    found = {}
    for key, label in labels.items():
        row = re.search(
            r"^\|\s*" + re.escape(label) + r"\s*\|\s*([+-]?[\d.]+)\s*\|\s*$",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
        if row:
            found[key] = float(row.group(1))
    return found


def r_quotes_the_engine(text: str) -> dict[str, str]:
    """What the vignette states the engine's forty-epoch run measured, AS WRITTEN.

    The strings rather than the floats, because how many decimals a page quotes is part of
    what it claims: `0.978` is a claim about three of them and nothing about the fourth.
    """
    said = re.search(
        # The digits and not a trailing full stop: `[\d.]+` swallowed the one that ends
        # the sentence and handed `float()` the string `+0.004.`, which is §4's extractor
        # answering a question one character wider than the one asked.
        r"correlation of\s+(\d+(?:\.\d+)?)\s+and a mean difference of\s+([+-]?\d+(?:\.\d+)?)",
        re.sub(r"\s+", " ", text),
    )
    if not said:
        return {}
    return {"correlation": said.group(1), "mean_difference": said.group(2)}


def article_quotes_r(text: str) -> tuple[str, float | None]:
    """What the article says about the other page's gap: a direction, and a number or not.

    The emphasis is stripped first because this one phrase has carried `about five\npoints
    *below*` and `**several points below**` inside one month - the markdown around a word is
    not part of the claim, and neither is where the line happens to wrap.
    """
    flat = re.sub(r"\s+", " ", re.sub(r"[*_]", "", text))
    said = re.search(
        r"clDice sits (?:about )?([a-z]+|\d+(?:\.\d+)?) points? (below|above) Dice", flat
    )
    if not said:
        return "", None
    stated, direction = said.group(1), said.group(2)
    if re.fullmatch(r"\d+(?:\.\d+)?", stated):
        return direction, float(stated)
    return direction, NUMBER_WORDS.get(stated)


@pytest.fixture(scope="module")
def retinal_pair():
    """Both pages, with the same control the section walk uses rather than a second one."""
    theirs = fetch(VIGNETTE.format(name=RETINAL))
    if theirs is None:
        if fetch(CONTROL) is not None:
            pytest.fail(
                f"{CONTROL} is reachable and vignettes/{RETINAL}.Rmd is not - the vignette "
                "moved or was renamed, which is a finding rather than a network problem"
            )
        pytest.skip(f"no network: vignettes/{RETINAL}.Rmd could not be fetched")
    return RETINAL_ARTICLE.read_text(), theirs


def test_the_retinal_pages_still_print_the_figures_they_argue_from(retinal_pair):
    """The positive control, and the half that turns a rewrite into a failure.

    Without it the two assertions below compare empty dicts and pass - §4's category in the
    form where both sides agree on nothing, and the wrap's-warning pair's own lesson: assert
    that each half was found before comparing them. So a page rewritten past these extractors
    fails as *the page changed* rather than quietly holding nothing.
    """
    ours, theirs = retinal_pair

    mine = article_table_figures(ours)
    assert {"correlation", "mean_difference"} <= set(mine), (
        f"{RETINAL_ARTICLE.name} no longer prints its own clDice-minus-Dice table rows - "
        f"found {sorted(mine)}. The page was rewritten, or the row labels moved"
    )

    printed = r_printed_figures(theirs)
    assert {"correlation", "mean_difference"} <= set(printed), (
        f"vignettes/{RETINAL}.Rmd no longer prints `correlation` and `mean_difference` as a "
        f"named vector - found {sorted(printed)}. The chunk changed, or it stopped being knit"
    )

    assert r_quotes_the_engine(theirs), (
        f"vignettes/{RETINAL}.Rmd stopped quoting the engine's forty-epoch correlation and "
        "mean difference. If that cross-reference went on purpose, this pair is smaller than "
        "it was and the §3F row says so; if not, the page lost the evidence for its claim"
    )

    direction, _ = article_quotes_r(ours)
    assert direction, (
        f"{RETINAL_ARTICLE.name} stopped saying which way the vignette's gap runs. That "
        "sentence is the article's half of the shared claim, so losing it loses the argument "
        "rather than only the number"
    )


def page_states_its_own_sign(text: str) -> str:
    """Which way a page says ITS OWN mean difference runs, read out of its own sentence.

    Hard-coding the two signs looked right and was the weaker test: `forty >= 0` beside
    `fifteen < 0` is an invented pair of bounds, and a forty-epoch rebuild landing at -0.001
    would fail it on a page that had been rewritten to say so. Read from the prose, the same
    assertion holds the figure to the sentence over it, which is a pair rather than a guess -
    and it leaves the ordering below with something of its own to say.

    Exactly one of the two words, or nothing: prose carrying both is prose this extractor
    cannot read, and answering anyway would be §4's probe giving the comfortable answer.
    """
    flat = re.sub(r"\s+", " ", re.sub(r"[*_]", "", text))
    said = re.search(r"mean difference is ([^.;\-]+)", flat, re.IGNORECASE)
    if not said:
        return ""
    claimed = {word for word in ("positive", "negative") if word in said.group(1).lower()}
    return claimed.pop() if len(claimed) == 1 else ""


def test_each_retinal_page_agrees_with_its_own_PROSE_about_the_sign_of_its_gap(retinal_pair):
    """The sign, held to the sentence each page writes over its own figure.

    Two pairs rather than two bounds. The article says the mean difference is *small and
    positive* and measures one; the vignette says *negative and not small* and measures one.
    A re-knit that moves a figure across zero without touching the sentence fails here, which
    is the half of `#415` that lives inside one page rather than between the two.
    """
    for label, text, figures in (
        (RETINAL_ARTICLE.name, retinal_pair[0], article_table_figures(retinal_pair[0])),
        (f"vignettes/{RETINAL}.Rmd", retinal_pair[1], r_printed_figures(retinal_pair[1])),
    ):
        claimed = page_states_its_own_sign(text)
        assert claimed, (
            f"{label} no longer says which way its own mean difference runs, or says both - "
            "the sentence is what the figure under it is held to, so this is not a pass"
        )
        measured = figures["mean_difference"]
        assert (measured >= 0) == (claimed == "positive"), (
            f"{label} says its mean clDice-minus-Dice is {claimed} and measures "
            f"{measured:+.4f}. The figure was rebuilt and the sentence over it was not"
        )


def test_the_better_trained_retinal_run_closes_the_gap(retinal_pair):
    """The shared claim, and the only thing the two pages assert together.

    *The better-trained model closes the gap* is written on both pages, in those words, and
    neither page measures it: the article has forty epochs and the vignette fifteen. Held as
    an ordering rather than as either number, because both numbers move whenever either page
    is rebuilt - the vignette's did, which is what `#415` is - and because the ordering is the
    sentence, where the two signs are only today's way of satisfying it.

    It is its own test rather than a third assertion beside the signs: in one test the sign
    assertions run first and a mutation that moves a figure far enough to break the ordering
    breaks a sign too, so the line below would never be reached and would be guarded by
    nothing. §4's *ask what the change would have to break* - here the answer was *nothing*,
    inside a test that was passing.
    """
    ours, theirs = retinal_pair
    forty = article_table_figures(ours)["mean_difference"]
    fifteen = r_printed_figures(theirs)["mean_difference"]
    assert forty > fifteen, (
        f"the forty-epoch gap is {forty:+.4f} and the fifteen-epoch one is {fifteen:+.4f}, so "
        "the better-trained model does NOT close the gap on these figures. That sentence is "
        "written on both pages and is the reason either of them quotes the other"
    )


def test_each_retinal_page_quotes_the_others_figures_as_that_page_prints_them(retinal_pair):
    """A number quoted of the other page is held to that page's own output; a word is not.

    The two halves run in opposite directions and that is the point. The vignette states the
    engine's figures to three decimals, so they are held to three. The article states only a
    direction for the vignette's, so only the direction is held - and if it ever names a count
    of points again, that count is held to within `POINT` of the vignette's own figure, which
    is what the stale *five* against -0.0833 failed.
    """
    ours, theirs = retinal_pair
    mine = article_table_figures(ours)
    printed = r_printed_figures(theirs)

    for key, literal in r_quotes_the_engine(theirs).items():
        places = len(literal.partition(".")[2])
        assert round(mine[key], places) == pytest.approx(float(literal), abs=10 ** -(places + 3)), (
            f"vignettes/{RETINAL}.Rmd tells a reader the engine's {key.replace('_', ' ')} is "
            f"{literal}, and {RETINAL_ARTICLE.name} measures {mine[key]:+.4f}. One of the two "
            "pages was rebuilt and the other's sentence about it was not"
        )

    direction, stated = article_quotes_r(ours)
    fifteen = printed["mean_difference"]
    assert (fifteen < 0) == (direction == "below"), (
        f"{RETINAL_ARTICLE.name} says clDice sits {direction} Dice in the vignette, and the "
        f"vignette measures {fifteen:+.4f}"
    )
    if stated is not None:
        points = abs(fifteen) * 100
        assert abs(stated - points) <= POINT, (
            f"{RETINAL_ARTICLE.name} pins the vignette's gap at {stated:g} points and the "
            f"vignette measures {points:.2f}. A number quoted of another page is a number that "
            "page can move: either re-read it, or say the direction without a count, which is "
            "what this sentence does today"
        )
