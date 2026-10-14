"""The issue forms and the pull-request template, in the shape both repositories share.

Same invariant as the README and the reference groups, and here for the same reason: the two
cannot be equal - only an R user can be unsure which half they are in, and only this side can
ask for a torch version - but they can ask the same things in the same order, so a contributor
who has filed in one repository is not starting again in the other.

Checked rather than left to care, because the templates these replace asked for a **TensorFlow
version**. Both of them did, in a project that has been on torch since it was rewritten; the R
one told the reporter to run `tensorflow::tf_version()`, which that package does not have. They
were written in 2022 and 2023 and nothing has read them since - the one document in the
repository that is addressed to somebody who has never seen the code.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml  # a core dependency, so `importorskip` here could only hide a broken install

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / ".github" / "ISSUE_TEMPLATE"
PULL_REQUEST = ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"

# Also in platypus's tools/check-github-templates.R. `half` is R's alone - a user here is
# already in the engine - so each side asserts a subsequence of this, never equality.
BUG_FIELD_ORDER = [
    "what-happened",
    "half",
    "expected",
    "reproduce",
    "session",
    "anything-else",
]

FEATURE_FIELD_ORDER = [
    "problem",
    "proposal",
    "alternatives",
    "measurement",
    "anything-else",
]

# The subsequence assertions permit absence, which is how drift gets back in, so the fields
# that must be present are named separately. `session` is the whole point of the exercise.
BUG_REQUIRED = ["what-happened", "expected", "reproduce", "session"]
FEATURE_REQUIRED = ["problem", "proposal"]

PULL_REQUEST_HEADINGS = ["What this changes", "Closes", "How it was checked"]


def field_ids(name: str) -> list[str]:
    form = yaml.safe_load((TEMPLATES / name).read_text())
    return [block["id"] for block in form["body"] if "id" in block]


def required_ids(name: str) -> set[str]:
    form = yaml.safe_load((TEMPLATES / name).read_text())
    return {
        block["id"]
        for block in form["body"]
        if block.get("validations", {}).get("required") and "id" in block
    }


@pytest.mark.parametrize(
    ("form", "order"),
    [("bug_report.yml", BUG_FIELD_ORDER), ("feature_request.yml", FEATURE_FIELD_ORDER)],
)
def test_the_fields_are_a_subsequence_of_the_shared_order(form, order):
    got = field_ids(form)

    assert got, f"{form} has no identified fields"
    assert not set(got) - set(order), (
        f"{form} has fields outside the shared order: {sorted(set(got) - set(order))}. "
        "Adding one means adding it here and in platypus's tools/check-github-templates.R"
    )
    assert got == [one for one in order if one in got], (
        f"{form} is out of the order both repositories share: {got}"
    )


@pytest.mark.parametrize(
    ("form", "must_have"),
    [("bug_report.yml", BUG_REQUIRED), ("feature_request.yml", FEATURE_REQUIRED)],
)
def test_the_fields_that_must_be_asked_are_asked(form, must_have):
    """The subsequence test permits absence; this is the half that does not."""
    got = set(field_ids(form))

    assert not set(must_have) - got, f"{form} stopped asking: {sorted(set(must_have) - got)}"


def test_the_version_question_cannot_be_skipped():
    """A markdown section gets deleted. `required: true` does not, which is why these are forms.

    And the prefilled commands have to name this package, or the reporter pastes a Python
    version and nothing else - which is what the old template collected.
    """
    assert "session" in required_ids("bug_report.yml")

    form = yaml.safe_load((TEMPLATES / "bug_report.yml").read_text())
    session = next(b for b in form["body"] if b.get("id") == "session")
    prefilled = session["attributes"]["value"]

    assert "pyplatypus.__version__" in prefilled
    assert "torch" in prefilled


def test_no_template_asks_for_tensorflow():
    """The regression. Both repositories asked for it until 2026-10-11.

    What is searched is the **questions** - each form's `body` - not the raw file and not
    `config.yml`. Two calibrations were wrong before this one: reading the raw bytes failed on
    the comment in `bug_report.yml` that records this very mistake, and searching the whole
    parsed file flagged a contact link in the R package's `config.yml` that points at the 2020
    TensorFlow package on purpose, which is where its stars and open issues still are. Sending
    somebody there is the opposite of the error. A rule aimed at a word rather than at the
    error catches the documentation of the error.
    """
    offenders = []
    for path in sorted(TEMPLATES.glob("*_re*.yml")):
        asked = yaml.safe_dump(yaml.safe_load(path.read_text())["body"])
        if "tensorflow" in asked.lower():
            offenders.append(path.name)

    markdown = re.sub(r"<!--.*?-->", "", PULL_REQUEST.read_text(), flags=re.DOTALL)
    if "tensorflow" in markdown.lower():
        offenders.append(PULL_REQUEST.name)

    assert not offenders, f"these still ask about TensorFlow: {offenders}"


def test_the_reader_is_told_where_the_other_half_lives():
    """§3G routes issues by where the change is visible, so the wrong repository is the
    predictable failure, and a contact link is the only place GitHub shows that choice."""
    config = yaml.safe_load((TEMPLATES / "config.yml").read_text())
    links = config.get("contact_links") or []

    assert any("maju116/rplatypus" in link.get("url", "") for link in links), (
        "no contact link points at the R package"
    )


def test_the_pull_request_template_shares_its_headings():
    text = PULL_REQUEST.read_text()
    headings = [line[3:].strip() for line in text.splitlines() if line.startswith("## ")]

    assert headings == PULL_REQUEST_HEADINGS, (
        f"the shared headings changed on this side only: {headings}"
    )
    assert "mutated to prove it can fail" in text, (
        "the one check in here that is this project's own, and the first to be dropped"
    )
