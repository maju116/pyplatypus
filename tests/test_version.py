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
