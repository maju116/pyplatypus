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
RUNNER = re.compile(r"^(ubuntu|macos|windows)-[a-z0-9.]+$")


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
    """Every runner family the suite's own workflow names, found wherever it names them.

    Walked rather than read off `runs-on`, because the matrix job's `runs-on` is
    `${{ matrix.os }}` and the labels are in `strategy.matrix.os` and in its `include` -
    which is three places a label can hide and two of them are not the obvious one.
    """
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            match = RUNNER.match(node.strip())
            if match:
                found.add(match.group(1))

    walk(yaml.safe_load(CI.read_text(encoding="utf-8")))
    return found


def test_everything_below_was_read_at_all():
    """The control, because every assertion in this file is over something parsed.

    An empty section, an unreadable manifest or a walk that found no runner labels would
    make the comparisons below pass over nothing, which is §4's count over no input.
    """
    section = installing_section()
    assert len(section) > 500, f"the Installing section is {len(section)} characters"
    assert "pip install" in section, "the Installing section installs nothing"

    families = ci_runner_families()
    assert len(families) >= 2, f"runner families found in ci.yml: {sorted(families)}"
    assert families <= set(RUNNER_CLASSIFIERS), f"unmapped runner family: {sorted(families)}"


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
    """
    classifiers = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["classifiers"]
    claimed = {c for c in classifiers if c.startswith("Operating System ::")}
    expected = {RUNNER_CLASSIFIERS[family] for family in ci_runner_families()}

    assert claimed == expected, (
        f"the manifest claims {sorted(claimed)} and ci.yml runs {sorted(expected)}"
    )
