"""The spec layer validates a configuration and nothing else.

`common.py` has said so since it was written - "kept here rather than beside the reader so
that the spec stays free of numpy and pydicom" - and nothing checked it. It matters because
validating a configuration is the one thing that happens before anything is read: R does it
to tell a user their file is wrong, and a test does it to assert a refusal. Pulling numpy,
torch, pydicom or albumentations into that path makes the cheapest operation in the package
depend on the most expensive part of the install.

Read from the source rather than from `sys.modules`, because `import pyplatypus.spec`
executes `pyplatypus/__init__.py` first, which imports the engine - so asking the import
system this question always answers "yes, numpy is loaded" whatever the spec layer does.
The first version of this did exactly that and reported a violation that was not there.
"""

import ast
from pathlib import Path

import pytest

HEAVY = {
    "numpy",
    "torch",
    "pydicom",
    "albumentations",
    "nibabel",
    "PIL",
    "timm",
    "safetensors",
    "huggingface_hub",
}

LAYER = sorted(Path("pyplatypus/spec").glob("*.py"))


def module_level_imports(path: Path) -> set[str]:
    """Top-level imports only: a deferred one inside a function is the intended escape."""
    tree = ast.parse(path.read_text())
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_there_are_modules_to_check():
    """A glob that matched nothing would make every assertion below pass."""
    assert len(LAYER) >= 7


@pytest.mark.parametrize("path", LAYER, ids=lambda p: p.name)
def test_no_module_level_dependency_on_the_data_stack(path):
    offenders = module_level_imports(path) & HEAVY
    assert not offenders, (
        f"{path} imports {', '.join(sorted(offenders))} at module level. Validating a "
        f"configuration happens before anything is read; import it inside the function "
        f"that needs it, as `detection.py` does with STRIDES."
    )


@pytest.mark.parametrize("path", LAYER, ids=lambda p: p.name)
def test_the_spec_does_not_reach_into_the_rest_of_the_package(path):
    """Except where it is deliberate and deferred. `pyplatypus.errors` is fine - it is the
    exception types and imports nothing."""
    reached = {name for name in module_level_imports(path) if name == "pyplatypus"}
    if not reached:
        return
    tree = ast.parse(path.read_text())
    modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pyplatypus")
    }
    allowed = {"pyplatypus.errors"}
    stray = {m for m in modules if not m.startswith("pyplatypus.spec") and m not in allowed}
    assert not stray, f"{path} imports {', '.join(sorted(stray))} at module level"
