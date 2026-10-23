"""Every function the schema's prose names is a function this package has.

The field descriptions in `pyplatypus/spec/` are not comments. `tools/build_config_reference.py`
renders them as the configuration reference, so a sentence there is published documentation and
a name in backticks reads as something to go and call.

Two of them were not. `Region.name` said its value is "what `save_class_masks` puts in a
filename" and `save_class_masks` existed in neither package - the name was decided on
`maju116/platypus#91` and written into the schema before anything was built, and then nothing
read the sentence again. `CsvLogger.path` promised "the same numbers `training_history()`
returns", which is the **R** package's function: true for the R reader who was in mind when it
was written, and a dead end for the Python reader the page is for.

This is §4's "an instruction addressed to a human is code nothing runs" in the one place the
project generates documentation from code. The engine's own suite is where it belongs, because
the schema is the engine's and the page is the engine's.

What this does not assert: that the prose is *good*, or that a name it does not mention should
be mentioned. Only that a name offered to a reader resolves.
"""

from __future__ import annotations

import re

import pytest

from pyplatypus import spec_schema

# Both of these live in `tests/conftest.py` now, because `test_articles.py` asks the same
# question of the articles' prose. `NOT_OURS` is listed rather than pattern-matched: a pattern
# for "looks like a builtin" is the loose calibration §4 warns about, and the list is short
# because the prose is written for a reader who already knows Python.
from tests.conftest import NOT_OURS, every_name_this_package_has


def descriptions_in(node) -> list[str]:
    """Every `description` string anywhere in the generated schema."""
    if isinstance(node, dict):
        found = [node["description"]] if isinstance(node.get("description"), str) else []
        for key, value in node.items():
            if key != "description":
                found += descriptions_in(value)
        return found
    if isinstance(node, list):
        return [text for item in node for text in descriptions_in(item)]
    return []


DESCRIPTIONS = descriptions_in(spec_schema())
MENTIONED = sorted(
    {
        name
        for text in DESCRIPTIONS
        for name in re.findall(r"`([a-z_][a-z0-9_]*)\(\)`", text)
        if name not in NOT_OURS
    }
)


def test_the_scan_found_the_schema_and_its_prose():
    """Or every assertion below passes by looking at nothing, which is how a guard written
    against exactly this mistake has itself made it in this project more than once."""
    assert len(DESCRIPTIONS) > 150
    assert len(MENTIONED) > 5


@pytest.mark.parametrize("name", MENTIONED)
def test_a_name_the_schema_offers_a_reader_resolves(name):
    assert name in every_name_this_package_has(), (
        f"the schema's prose names `{name}()` and this package has no such name. A reader of "
        "the configuration reference meets it as something to call. If it is the R package's, "
        "say what the engine itself answers with instead - the page is read by people who "
        "have no R."
    )
