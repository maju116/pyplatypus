"""Every configuration the guide shows must be one the engine accepts.

A reference page cannot contradict the code, because it is generated from it. A guide is
written by hand, so it can - and the failure is the expensive kind: somebody copies a block,
it is refused, and the documentation taught them a mistake. Writing this page produced three
of those before the prose was finished: `model_checkpoint` has no `mode`, it does require a
`path`, and `anchors: auto` is not a thing - `auto` belongs to `window`, and anchors are
fitted by being left out.

The headings are pinned too. The R package's guide carries the same ones, so a reader
moving between the two sites finds the same walk; neither repository can see the other, so
the list is written out in both suites with a comment saying so.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

from pyplatypus import ConfigError, from_dict

ROOT = pathlib.Path(__file__).resolve().parent.parent
GUIDE = ROOT / "guide.qmd"

# Also in platypus's tests/testthat/test-guide.R. Two copies, because the alternative is one
# repository reaching into the other.
SECTIONS = [
    "One file, one run",
    "Several models at once",
    "Dividing one folder",
    "Augmentation, callbacks, and the rest of a model",
    "Detection instead of segmentation",
    "Not training at all",
    "What is checked, and when",
    "The same run from code",
]


def configurations() -> list[tuple[int, str]]:
    """Every fenced yaml block that is a configuration, with the line it starts on.

    A `yaml` fence that does not carry a `task:` is something else - the editor's schema
    comment is one - and validating it would fail for a reason the page is not making a
    claim about.
    """
    text = GUIDE.read_text()
    found = []
    for match in re.finditer(r"^```yaml\n(.*?)^```", text, re.MULTILINE | re.DOTALL):
        if re.search(r"^task:", match.group(1), re.MULTILINE):
            found.append((text[: match.start()].count("\n") + 1, match.group(1)))
    return found


def test_the_guide_shows_configurations_at_all():
    """A guard against the extractor silently matching nothing, which would pass every test."""
    assert len(configurations()) >= 5, (
        f"found {len(configurations())} configurations in the guide - "
        "the fence pattern is wrong, or the page is"
    )


@pytest.mark.parametrize(
    ("line", "block"), configurations(), ids=[f"line-{line}" for line, _ in configurations()]
)
def test_every_configuration_in_the_guide_is_accepted(line, block):
    """`check_paths=False`: the paths in the guide are illustrative, the fields are not."""
    try:
        from_dict(yaml.safe_load(block), check_paths=False)
    except ConfigError as refusal:
        pytest.fail(f"the configuration at line {line} of guide.qmd is refused:\n{refusal}")


def test_the_guide_walks_the_agreed_sections():
    headings = re.findall(r"^## (.+)$", GUIDE.read_text(), re.MULTILINE)
    assert headings == SECTIONS, (
        "the guide's sections changed; platypus's guide carries the same list, "
        "so change both or neither"
    )
