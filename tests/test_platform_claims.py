"""What this package says about where it runs, held to what it can actually do.

`pyplatypus#459` is an Intel Mac that cannot install the package at all: torch has published
no macOS `x86_64` wheel since 2.2.2 and the floor here is 2.7, so on that platform nothing
satisfies the one requirement that matters. The report came back from a real machine, and
what it met was neither of the real reasons - its Python was below `requires-python`, most
published versions of this package predate that floor, so the resolver walked backwards and
began building the 2022 TensorFlow package's dependency tree from source.

**The issue asked for a platform marker and the measurement refused it**, which is why there
is none to test: a marker can only switch a requirement OFF, so
`torch>=2.7 ; platform_system != "Darwin" or platform_machine == "arm64"` resolves to zero
requirements on Intel macOS - `Resolved in 1ms`, exit 0 - and the package installs without
torch and fails at import instead. Refusing by name would mean depending on a package that
does not exist, on a name anybody may register. So no metadata reaches that user, and the
README is the whole of the answer.

That makes the README load-bearing rather than descriptive, which is what this file is for.
Three claims, each with a different reader:

    the two floors it quotes          against the manifest, which is where they are declared
    the macOS sentence               against nothing - it is the fact no resolver will say,
                                     so what is held is that the three parts of it are there
    the operating-system classifiers  against the matrix CI actually runs
    the PYTHON classifiers            against the same matrix, and their lowest against
                                      `requires-python` - which is the floor's only guard

The classifiers were `Operating System :: OS Independent` until `0.8.0a83`, which is true of
the wheel - `py3-none-any` - and false about whether the thing installs. The list has no
architecture axis, checked against PyPI's own page, so `MacOS` is as precise as a classifier
can be and the qualification has to live in prose.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"

# A GitHub runner label's family, which is as much as a classifier can express.
RUNNER_CLASSIFIERS = {
    "ubuntu": "Operating System :: POSIX :: Linux",
    "macos": "Operating System :: MacOS",
    "windows": "Operating System :: Microsoft :: Windows",
}
# `[a-z0-9._-]+` and not `[a-z0-9.]+`: the first pattern stopped at a SECOND hyphen, so
# `windows-11-arm`, `ubuntu-24.04-arm` and `macos-15-intel` - all real labels - were not
# matched at all, and the comparison below then reported that `ci.yml` runs no Windows
# while every Windows cell was present. Loud, and in the wrong direction.
RUNNER = re.compile(r"^(ubuntu|macos|windows)-[a-z0-9._-]+$")

# A per-minor Python classifier. The bare `Programming Language :: Python :: 3` is excluded
# on purpose: it is a claim about the major version and says nothing a matrix could answer.
PYTHON_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (3\.\d+)$")
# A version as the matrix must spell it - a STRING, which is not a formality. YAML reads an
# unquoted `3.10` as the float 3.1, measured: `yaml.safe_load("python: [3.10]")` is `[3.1]`.
# So an unquoted 3.10 cell would enter the matrix reading as a version that does not exist,
# in the one place this file is asked to keep 3.10 out of.
PYTHON_VERSION = re.compile(r"^3\.\d+$")


def installing_section() -> str:
    """The README's `## Installing` section, up to the next top-level heading."""
    parts = README.read_text(encoding="utf-8").split("## Installing", 1)
    assert len(parts) == 2, "the README has no Installing section"
    return parts[1].split("\n## ", 1)[0]


def declared_floors() -> tuple[str, str]:
    """The Python and torch floors as the manifest declares them, which is the source."""
    text = PYPROJECT.read_text(encoding="utf-8")
    python = re.findall(r'^requires-python\s*=\s*">=([\d.]+)"', text, re.MULTILINE)
    # Anchored on the whole line so that the `pascal` extra's `"torch>=2.7,<2.8"` is not
    # read as a second declaration of the same floor: that is a cap, and it lives in a list.
    torch = re.findall(r'^\s*"torch>=([\d.]+)",$', text, re.MULTILINE)
    assert len(python) == 1, f"requires-python declarations: {python}"
    assert len(torch) == 1, f"base torch floor declarations: {torch}"
    return python[0], torch[0]


def ci_runner_families() -> set[str]:
    """The runner families the SUITE runs on: the `test` job's own matrix, and only it.

    This walked the whole workflow until `pyplatypus#464`, and that is the difference
    between a guard and a guard in name only. `lint`, `floors` and `build` all run on
    `ubuntu-latest`, so a walk over every label in the file answers *Linux* whatever the
    matrix does - and the mutation that proves it is one line: with
    `os: [ubuntu-latest]` changed to `[macos-latest]`, the suite runs on Linux NOWHERE
    and all four tests here passed. The classifiers are a claim about where the package
    is TESTED, so the matrix is the only part of the file that can answer it.

    Three places a label hides in that job and two are not the obvious one:
    `strategy.matrix.os`, each `strategy.matrix.include` entry's `os`, and `runs-on`
    itself when it is a literal rather than `${{ matrix.os }}`.
    """
    workflow = yaml.safe_load(CI.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs") or {}
    # The control for this helper: a renamed job must not read as a matrix with no
    # platforms, which would make every comparison below pass over an empty set.
    assert "test" in jobs, f"ci.yml has no `test` job; it has {sorted(jobs)}"
    job = jobs["test"]

    labels: list[str] = []
    runs_on = job.get("runs-on")
    if isinstance(runs_on, str) and "${{" not in runs_on:
        labels.append(runs_on)
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    entry = matrix.get("os")
    labels.extend(entry if isinstance(entry, list) else [entry] if entry else [])
    for included in matrix.get("include") or []:
        if isinstance(included, dict) and included.get("os"):
            labels.append(included["os"])

    unreadable = [label for label in labels if not RUNNER.match(str(label).strip())]
    assert not unreadable, (
        f"runner labels this pattern cannot read: {unreadable} - widen RUNNER rather "
        "than letting a label go uncounted, which is how `windows-11-arm` once read as "
        "no Windows at all"
    )
    return {RUNNER.match(str(label).strip()).group(1) for label in labels}


def ci_test_job_pythons() -> set[str]:
    """The Python minors the SUITE runs, from the `test` job's matrix and only it.

    `ci_runner_families()`'s argument one axis over, and for the same reason: `lint`,
    `floors`, `build` and `docs` each pin a Python of their own, so a walk over the whole
    workflow answers about jobs that are not the suite. `pyplatypus#464` is that lesson paid
    on the operating-system half.

    Two places a version hides in that job: `strategy.matrix.python`, and each
    `strategy.matrix.include` entry's `python`.

    Every value must be a STRING matching `3.x`. That is the 3.10 guard in the one shape
    nobody would look for: YAML reads an unquoted `3.10` as the float 3.1, so a cell written
    without quotes would be a version that does not exist, compared against classifiers that
    could never match it, and the failure would name `3.1` rather than the mistake.
    """
    workflow = yaml.safe_load(CI.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs") or {}
    assert "test" in jobs, f"ci.yml has no `test` job; it has {sorted(jobs)}"
    job = jobs["test"]

    values: list[object] = []
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    entry = matrix.get("python")
    values.extend(entry if isinstance(entry, list) else [entry] if entry else [])
    for included in matrix.get("include") or []:
        if isinstance(included, dict) and included.get("python") is not None:
            values.append(included["python"])

    unquoted = [v for v in values if not isinstance(v, str)]
    assert not unquoted, (
        f"these matrix Python versions are not strings: {unquoted} - quote them. YAML reads "
        "an unquoted 3.10 as the float 3.1, so the version in the job's name and the version "
        "this test compares would be different things"
    )
    unreadable = [v for v in values if not PYTHON_VERSION.match(v)]
    assert not unreadable, f"matrix Python versions this pattern cannot read: {unreadable}"
    return set(values)


def declared_pythons() -> set[str]:
    """The per-minor Python classifiers, which are what a reader on PyPI is told."""
    classifiers = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["classifiers"]
    return {match.group(1) for match in (PYTHON_CLASSIFIER.match(c) for c in classifiers) if match}


def test_everything_below_was_read_at_all():
    """The control, because every assertion in this file is over something parsed.

    An empty section, an unreadable manifest or a matrix that yielded no runner labels
    would make the comparisons below pass over nothing, which is §4's count over no input.
    """
    section = installing_section()
    assert len(section) > 500, f"the Installing section is {len(section)} characters"
    assert "pip install" in section, "the Installing section installs nothing"

    families = ci_runner_families()
    assert len(families) >= 2, f"runner families in the test job's matrix: {sorted(families)}"
    assert families <= set(RUNNER_CLASSIFIERS), f"unmapped runner family: {sorted(families)}"

    # The same control one axis over: two empty sets agree about everything, so a matrix
    # whose `python` key was renamed, or a classifier list this pattern cannot read, must
    # fail here rather than making the comparison below pass over nothing.
    pythons = ci_test_job_pythons()
    assert len(pythons) >= 2, f"Python versions in the test job's matrix: {sorted(pythons)}"
    claimed = declared_pythons()
    assert len(claimed) >= 2, f"per-minor Python classifiers: {sorted(claimed)}"


def test_the_installing_section_quotes_the_manifest_s_own_floors():
    """Two numbers, two files, and until `0.8.0a83` nothing held them together.

    The README is what a reader meets first - on PyPI it IS the page - and a floor it quotes
    is a promise about what will resolve. The numbers are read out of the prose by their role
    rather than compared as a formatted string, so the sentence can be rewritten and only a
    wrong number fails: the retinal pair's own lesson, where a hard-coded phrasing would have
    passed a stale figure.

    Its own first run failed on prose that was correct, which is the cheapest available
    argument for reading a failure before believing it: the version pattern was greedy
    over the sentence's full stop. When a new check fails, suspect the instrument.
    """
    python_floor, torch_floor = declared_floors()
    section = installing_section()

    # `[\d.]+` is greedy over the sentence's own full stop, so this pattern ends on a
    # digit: the first run of this test read the README's `torch ≥ 2.7.` as a floor of
    # `2.7.` and failed on prose that was right.
    version = r"(\d+(?:\.\d+)*)"
    said_python = set(re.findall(rf"Python ≥ {version}", section))
    said_torch = set(re.findall(rf"torch ≥ {version}", section))

    assert said_python, "the Installing section states no Python floor"
    assert said_torch, "the Installing section states no torch floor"
    assert said_python == {python_floor}, (
        f"the README says Python >= {sorted(said_python)} and the manifest says {python_floor}"
    )
    assert said_torch == {torch_floor}, (
        f"the README says torch >= {sorted(said_torch)} and the manifest says {torch_floor}"
    )


def test_the_installing_section_names_the_platform_that_cannot_install_it():
    """The fact `pyplatypus#459` exists for, and the only place it can be stated.

    Three parts, because any one of them alone is useless to the person it is for: which
    macOS works, which does not, and the torch version the claim is pinned to - so that a
    reader can check it themselves and a later reader can see what it was true of.
    """
    section = installing_section()
    missing = [part for part in ("Apple Silicon", "x86_64", "2.2.2") if part not in section]
    assert not missing, (
        f"the Installing section no longer says {missing} - which is the whole of what "
        "reaches an Intel Mac, since no metadata in this package can say it"
    )


def test_the_operating_system_classifiers_are_the_ones_ci_runs():
    """`These three are what CI runs` is a sentence in the manifest; this is what makes it one.

    Equality in both directions: a classifier naming an operating system nothing is run on is
    a claim, and a platform the matrix tests without a classifier is a reader not being told.
    `OS Independent` fails this by being neither.

    **And `ci.yml runs` means the SUITE's matrix**, which is `pyplatypus#464`: until then
    this compared against every runner label in the workflow, and `lint`, `floors` and
    `build` are all on `ubuntu-latest` - so the matrix could leave Linux entirely with this
    test still green. `ci_runner_families()` reads the `test` job and says why.
    """
    classifiers = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["classifiers"]
    claimed = {c for c in classifiers if c.startswith("Operating System ::")}
    expected = {RUNNER_CLASSIFIERS[family] for family in ci_runner_families()}

    assert claimed == expected, (
        f"the manifest claims {sorted(claimed)} and ci.yml runs {sorted(expected)}"
    )


def test_the_python_classifiers_are_the_ones_ci_runs():
    """The operating-system pair's argument one axis over, and it had no guard at all.

    `pyplatypus#470`: the OS classifiers have been held to the matrix in both directions
    since `0.8.0a83`, and the Python ones were held by nothing - so the claim went **two
    minors behind** what exists without anything failing. 3.14 was released 2025-10-07 and
    3.15 on 2026-10-09; this package ran end to end on 3.14.7 on a borrowed M3, which is
    where `#468` came from, and said nothing about it anywhere.

    Equality in both directions, for the OS half's reasons: a classifier for a version
    nothing is run on is a claim, and a version the matrix tests without a classifier is a
    reader not being told.
    """
    claimed = declared_pythons()
    tested = ci_test_job_pythons()

    assert claimed == tested, (
        f"the manifest claims Python {sorted(claimed)} and ci.yml's test job runs {sorted(tested)}"
    )


def test_the_lowest_python_anybody_claims_or_tests_is_the_declared_floor():
    """The floor's only guard, and the one that keeps 3.10 out.

    Python 3.10 is out of support and the floor moved from 3.10 to 3.11 with
    `requires-python`. That move cost three renamed required contexts, because a job's name
    is what branch protection matches - so the expensive half of removing a version is not
    the removal. With this assertion a 3.10 cell or a 3.10 classifier cannot come back
    without `requires-python` moving in the same breath, which is a line nobody edits by
    accident.

    Asked of the minimum rather than by naming 3.10: a test that forbids one version is a
    test that goes stale the next time the floor moves, and this one holds whatever the floor
    becomes. `3.11` goes out of support on 2027-10-31, and on that day this test is what
    says the matrix and the classifiers still agree with the manifest.
    """
    floor, _torch = declared_floors()
    both = declared_pythons() | ci_test_job_pythons()
    assert both, "no Python version was read from either the manifest or the matrix"

    def key(version: str) -> tuple[int, ...]:
        return tuple(int(part) for part in version.split("."))

    lowest = min(both, key=key)
    assert lowest == floor, (
        f"the lowest Python claimed or tested is {lowest} and `requires-python` declares "
        f">={floor} - one of the two moved without the other"
    )
