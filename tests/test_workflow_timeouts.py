"""Every job in every workflow carries a ceiling, which is `pyplatypus#457`.

Without `timeout-minutes` a hung job runs to GitHub's 360-minute default, and the sixteenth
review pass saw the suite deadlock once at a two-worker detection test: the main process
asleep, both children at zero CPU, `utime` unmoved across two reads thirteen minutes apart.
Nothing in either repository set a timeout, so that run would have burned six hours before
going red.

This is the COMPLETENESS half rather than a check of the figure. A ceiling nobody set is how
this comes back - a new job arrives and inherits the default silently - and the one thing a
test can do about it is refuse the job that has none. What the right number is cannot be
tested, only measured, and the figure's own argument lives in the workflow's comment beside
it.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _workflow_files() -> list[pathlib.Path]:
    return sorted(p for p in WORKFLOWS.glob("*.y*ml"))


def test_the_scan_finds_the_workflows_at_all():
    """The control. An empty glob would make every assertion below pass over nothing, which
    is the shape §4 calls a count over no input."""
    files = _workflow_files()
    assert len(files) >= 3, [p.name for p in files]
    names = {p.name for p in files}
    assert {"ci.yml", "publish.yml"} <= names, sorted(names)


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_job_has_a_timeout(path: pathlib.Path):
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs") or {}
    # The control for this file: a workflow with no jobs would pass the loop silently.
    assert jobs, path.name
    missing = [name for name, job in jobs.items() if "timeout-minutes" not in job]
    assert not missing, (
        f"{path.name}: these jobs would inherit GitHub's 360-minute default: {missing}"
    )
    # And a ceiling has to be a number somebody could reach, not a token: the longest job
    # either repository has had is 11.8 minutes, and §7 records one 22m55s step.
    for name, job in jobs.items():
        ceiling = job["timeout-minutes"]
        assert isinstance(ceiling, int), (name, ceiling)
        assert 20 <= ceiling <= 120, (name, ceiling)
