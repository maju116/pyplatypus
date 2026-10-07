"""The configuration page must name every field a configuration accepts.

The page is generated, so it cannot contradict the schema about what a field *means*. What it
can do is leave a block out: `split:` and `augmentation:` are reachable from nowhere else on
it, and the first version did omit them while describing 195 of the 197 fields and looking
finished. So the assertion is against the schema's own reachable definitions rather than
against a count - a count is what §4as's coverage guard used, and it passed with a third of
the tree invisible to it.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

from pyplatypus import spec_schema

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "tools" / "build_config_reference.py"


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    """The page as the docs build makes it.

    Generated into the repository because that is where the build puts it and where the
    renderer looks; it is gitignored, so this leaves nothing behind that git would show.
    """
    subprocess.run([sys.executable, str(GENERATOR)], check=True, capture_output=True)
    return (ROOT / "config.qmd").read_text()


def reachable_blocks() -> dict[str, list[str]]:
    """Every definition with fields that a configuration can reach, and its field names."""
    schema = spec_schema()
    defs = schema["$defs"]

    def refs_in(node) -> list[str]:
        if isinstance(node, dict):
            found = []
            if "$ref" in node:
                found.append(node["$ref"].rsplit("/", 1)[-1])
            for value in node.values():
                found += refs_in(value)
            return found
        if isinstance(node, list):
            return [name for item in node for name in refs_in(item)]
        return []

    seen: dict[str, list[str]] = {}
    queue = refs_in({k: v for k, v in schema.items() if k != "$defs"})
    while queue:
        name = queue.pop()
        if name in seen or name not in defs:
            continue
        definition = defs[name]
        fields = list(definition.get("properties", {}))
        if fields:
            seen[name] = fields
        queue += refs_in(definition)
    return seen


def test_every_block_is_described(page):
    """A block missing from the page is a block a reader cannot look up."""
    blocks = reachable_blocks()
    assert len(blocks) > 30, f"only {len(blocks)} blocks found - the walk is wrong, not the page"

    missing = []
    for name, fields in sorted(blocks.items()):
        # A block appears as a definition list of its own fields - one line holding just the
        # name in backticks, then a `:` line. Checking every field of the block is what says
        # the list is complete rather than started.
        if not all(re.search(rf"^`{re.escape(field)}`$", page, re.MULTILINE) for field in fields):
            missing.append(name)
    assert not missing, f"blocks with fields missing from the page: {missing}"


def test_every_field_is_named(page):
    """Every field name in the schema, including the ones only a nested block has."""
    blocks = reachable_blocks()
    names = {field for fields in blocks.values() for field in fields}
    listed = set(re.findall(r"^`([a-z_0-9]+)`$", page, re.MULTILINE))

    assert names, "no field names in the schema - the walk is wrong"
    assert not (names - listed), f"in the schema, not on the page: {sorted(names - listed)}"


def test_the_editor_schema_is_written_and_self_consistent(page):
    """The page tells an editor where the schema is, and that is where the build puts it."""
    schema = spec_schema()
    written = ROOT / "schema" / "spec.schema.json"

    assert written.exists(), "the generator writes the schema beside the page"
    assert f"$schema={schema['$id']}" in page, "the page must quote the schema's own $id"
    assert schema["$id"].startswith("https://maju116.github.io/pyplatypus/"), (
        "the $id must be on the site that publishes it - see tools/build_config_reference.py"
    )


def test_every_internal_link_has_an_explicit_anchor(page):
    """A link to a heading must not depend on how the renderer derives an id from it.

    It did once, and the link went nowhere: pandoc keeps the underscore in
    `## `task: semantic_segmentation``, so a link written as
    `#task-semantic-segmentation` had no target - measured in the rendered HTML, not reasoned
    about. Every cross-link on the page now points at an `{#id}` written beside the heading,
    which is the same string, so this test needs no renderer to be true.
    """
    targets = set(re.findall(r"\]\(#([^)]+)\)", page))
    anchors = set(re.findall(r"\{#([^}]+)\}", page))

    assert targets, "no internal links on the page at all - the generator stopped writing them"
    assert anchors, "no explicit anchors - a renderer's id derivation is being trusted again"
    assert not (targets - anchors), f"links with no explicit anchor: {sorted(targets - anchors)}"


def test_no_two_headings_collide(page):
    """`dice`, `iou` and `tversky` are each both a loss and a metric, so their headings repeat.

    **The first version of this test read the explicit anchors and was worthless**: removing
    them from the generator left no anchors to find, no duplicates among none, and a pass -
    the test calibrated to admit the exact failure it was written to prevent. It reads the
    headings now, which exist whether or not anybody remembered an anchor, and a repeated
    heading is only acceptable when each one carries an id of its own.
    """
    headings = re.findall(r"^#{2,4} (.+?)(?: \{#([^}]+)\})?$", page, re.MULTILINE)

    assert len(headings) > 30, f"only {len(headings)} headings - the pattern is wrong, not the page"

    seen: dict[str, list[str | None]] = {}
    for text, anchor in headings:
        seen.setdefault(text, []).append(anchor or None)

    unanchored = {text: ids for text, ids in seen.items() if len(ids) > 1 and None in ids}
    assert not unanchored, (
        f"headings repeated with no explicit anchor, so the renderer will invent one: "
        f"{sorted(unanchored)}"
    )

    anchors = [a for _, a in headings if a]
    duplicates = {a for a in anchors if anchors.count(a) > 1}
    assert not duplicates, f"two headings with one anchor: {sorted(duplicates)}"
