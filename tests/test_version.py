"""The version lives in two places, and nothing used to check they agree.

`0.3.0a14` shipped to PyPI with `pyproject.toml` saying `0.3.0a14` and
`pyplatypus.__version__` still saying `0.3.0a13`, because the release bumped one and not
the other. Nothing failed: the wheel installed, the metadata was right, the tests passed,
and the only visible symptom was the R package's bridge refusing to believe the engine was
the version it had pinned.

It is worse than a cosmetic mismatch, because `__version__` is what gets *written down*:
`write_record()` stamps it into every run record and `export_weights()` into every weights
sidecar as `trained_with`. A wrong value there is wrong provenance, in files that outlive
the release.
"""

import re
from pathlib import Path

import pytest

import pyplatypus

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def declared_in_pyproject() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml has no top-level version"
    return match.group(1)


def test_the_package_reports_the_version_it_is_built_as():
    """The check that would have caught the 0.3.0a14 release before it was a release."""
    assert pyplatypus.__version__ == declared_in_pyproject()


def test_the_installed_distribution_agrees_too():
    """And the metadata, when the distribution is installed rather than merely importable.

    Skipped rather than failed when it is not: running the tests against a source tree is
    a thing people do, and it is not what this is about.
    """
    metadata = pytest.importorskip("importlib.metadata")
    try:
        installed = metadata.version("pyplatypus")
    except metadata.PackageNotFoundError:
        pytest.skip("pyplatypus is importable but not installed as a distribution")
    assert installed == pyplatypus.__version__


def test_the_citation_carries_the_version_being_released():
    """`CITATION.cff` states a version, and a citation naming the wrong one is worse than none.

    It is a committed file that no release step reads, so nothing else would notice it going
    stale - the same shape as `__version__` and `pyproject.toml` drifting apart, which shipped
    `0.3.0a14` calling itself `0.3.0a13` and was caught only by the R package's bridge test.
    """
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    citation = yaml.safe_load((root / "CITATION.cff").read_text())

    assert citation["version"] == pyplatypus.__version__, (
        f"CITATION.cff says {citation['version']}, the package says {pyplatypus.__version__}"
    )


def test_the_citation_carries_the_date_the_changelog_gives_that_version():
    """`date-released` is the fifth place a release is written down and the only one nothing
    held. It was last moved for `0.8.0a10` and the engine released eleven more versions
    above it, which is the same drift `version` had before the test above - the file is
    committed, no release step reads it, and a citation dated eleven releases early is wrong
    in the one field a citation is for.

    Held to the changelog rather than to a clock: the date a release claims is the one its
    own entry claims, both are written by hand, and a test that compares them catches the
    release that bumps one and forgets the other. It cannot catch both being wrong together,
    which is the limit worth stating rather than papering over with `date.today()` - that
    would fail every time the suite is run on a day that is not release day.
    """
    import datetime
    import pathlib
    import re

    import yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    citation = yaml.safe_load((root / "CITATION.cff").read_text())
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")

    heading = re.search(
        rf"^## \[{re.escape(pyplatypus.__version__)}\] - (\d{{4}}-\d{{2}}-\d{{2}})\s*$",
        changelog,
        re.MULTILINE,
    )
    assert heading, (
        f"CHANGELOG.md has no dated `## [{pyplatypus.__version__}]` heading; a version being "
        "released needs its own section, and `## [Unreleased]` is not one"
    )
    released = citation["date-released"]
    if isinstance(released, str):
        released = datetime.date.fromisoformat(released)
    assert released == datetime.date.fromisoformat(heading.group(1)), (
        f"CITATION.cff is dated {released} and CHANGELOG.md dates "
        f"{pyplatypus.__version__} {heading.group(1)}"
    )
