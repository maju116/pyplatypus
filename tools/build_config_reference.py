"""Turn `spec_schema()` into the page that explains a configuration file.

The configuration is the one thing that is **identical** between the two packages: the YAML
an R user writes is the YAML a Python user writes, because both are read by this engine's
pydantic models. So it gets exactly one exhaustive reference, generated from the schema those
models produce, and the R package's site links to it rather than keeping a second copy - which
is what §4aa of the project notes spent a day removing elsewhere.

Generated at docs-build time rather than committed. `_quarto.yml` is committed because two
consumers have to agree about it; this page has one consumer, the renderer, so there is
nothing for a staleness test to catch that the build would not.

Reading the schema rather than walking the models is deliberate, and it is the §4as lesson:
a hand-rolled traversal missed every `Annotated` field - 25 properties across 11 models - and
the guard written to notice that passed anyway, because it counted. pydantic generates the
schema, so no traversal of mine can be wrong about what is in it.
"""

from __future__ import annotations

import json
import pathlib
import textwrap

from pyplatypus import spec_schema

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "config.qmd"
SCHEMA_OUT = ROOT / "schema" / "spec.schema.json"

# The families, in the order a reader meets them in a specification. Each is a discriminated
# union in the schema, so the variants are found rather than listed here.
FAMILIES = [
    ("Losses", "loss", "SegmentationModel"),
    ("Metrics", "metrics", "SegmentationModel"),
    ("Optimisers", "optimizer", "SegmentationModel"),
    ("Callbacks", "callbacks", "SegmentationModel"),
]


def _unwrap(node: dict) -> dict:
    """A field's schema with pydantic's optionality wrapper removed.

    An optional field is `anyOf: [<the type>, {type: null}]`, which says nothing a reader
    wants in a type column - the default column already says whether it may be left out.
    """
    if "allOf" in node and len(node["allOf"]) == 1:
        node = {**node, **node["allOf"][0]}
        node.pop("allOf", None)
    options = node.get("anyOf")
    if not options:
        return node
    real = [o for o in options if o.get("type") != "null"]
    if len(real) == 1:
        merged = {**{k: v for k, v in node.items() if k != "anyOf"}, **real[0]}
        return merged
    return node


def _ref_name(node: dict) -> str | None:
    ref = node.get("$ref")
    return ref.rsplit("/", 1)[-1] if ref else None


def _plural(described: str) -> str:
    """`a list of a number` is not English. Only the scalars need this; a union is already
    phrased as "each one of", and a block name is already plural at its call site."""
    return {
        "a number": "numbers",
        "a whole number": "whole numbers",
        "text": "text values",
        "a mapping": "mappings",
        "a list": "lists",
        "`true` or `false`": "booleans",
    }.get(described, described)


def _task_anchor(tag: str) -> str:
    """One expression for the heading's id and for the link to it.

    Guessing how the renderer derives an id from a heading got this wrong: pandoc keeps the
    underscore in `task: semantic_segmentation`, and the link written as
    `#task-semantic-segmentation` went nowhere. An explicit anchor cannot disagree with the
    link, because they are the same string.
    """
    return f"task-{tag}"


def _discriminator_tags(node: dict) -> list[str]:
    """The `name:` values a discriminated union accepts, in the order it declares them."""
    mapping = (node or {}).get("discriminator", {}).get("mapping")
    return list(mapping) if mapping else []


def describe_type(node: dict, defs: dict) -> str:
    """What to write in this field, in the words a YAML writer uses.

    A discriminated union is named by its `name:` tags rather than by the pydantic classes
    behind them, because the tag is what gets typed. Those unions carry the discriminator on
    the **field** rather than behind a `$ref` - measured: `loss`, `metrics`, `optimizer` and
    `callbacks` all do - and reading only the `$ref` left them as "any", which is a type cell
    that tells a reader nothing.
    """
    node = _unwrap(node)

    tags = _discriminator_tags(node)
    if tags:
        return "one of " + ", ".join(f"`{tag}`" for tag in tags)

    name = _ref_name(node)
    if name:
        target = defs.get(name, {})
        if "enum" in target:
            return " or ".join(f"`{value}`" for value in target["enum"])
        tags = _discriminator_tags(target)
        if tags:
            return "one of " + ", ".join(f"`{tag}`" for tag in tags)
        return f"a `{name}` block"

    if "enum" in node:
        return " or ".join(f"`{value}`" for value in node["enum"])

    if node.get("anyOf"):
        return " or ".join(describe_type(option, defs) for option in node["anyOf"])

    kind = node.get("type")
    if kind == "array":
        inner = _unwrap(node.get("items", {}))
        if not inner:
            return "a list"
        inner_tags = _discriminator_tags(inner) or _discriminator_tags(
            defs.get(_ref_name(inner) or "", {})
        )
        if inner_tags:
            return "a list, each one of " + ", ".join(f"`{tag}`" for tag in inner_tags)
        inner_name = _ref_name(inner)
        if inner_name and not defs.get(inner_name, {}).get("enum"):
            return f"a list of `{inner_name}` blocks"
        return f"a list of {_plural(describe_type(inner, defs))}"
    if kind == "object":
        return "a mapping"
    return {
        "string": "text",
        "integer": "a whole number",
        "number": "a number",
        "boolean": "`true` or `false`",
        "null": "`null`",
    }.get(kind, kind or "any")


def describe_default(node: dict, name: str, required: list[str]) -> str:
    if name in required:
        return "**required**"
    if "default" not in node:
        # No default and not required: pydantic omits a `default_factory`, so there is no
        # value to quote and inventing one would be a claim about behaviour.
        return "optional"
    value = node["default"]
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, str):
        return f"`{value}`" if value else '`""`'
    return f"`{json.dumps(value)}`"


def field_table(definition: dict, defs: dict) -> list[str]:
    """One field per definition-list entry, rather than per table row.

    A table was the first shape and measuring the descriptions killed it: the median is 202
    characters, the ninth decile 383, **four carry fenced code blocks** - which do not render
    inside a `|` cell at all - and one carries a `|` of its own, which silently splits the
    cell it lands in. A definition list renders every one of those intact, keeps the markdown
    the descriptions were written in, and adds nothing to the page's table of contents, which
    206 sub-headings would have flooded.
    """
    required = definition.get("required", [])
    rows: list[str] = []
    for name, node in definition.get("properties", {}).items():
        unwrapped = _unwrap(node)
        kind = describe_type(node, defs)
        default = describe_default(unwrapped, name, required)
        rows += [f"`{name}`", f":   {kind} — {default}", ""]
        description = (unwrapped.get("description") or "").strip()
        if description:
            # Four spaces, so every paragraph and fence stays inside the definition rather
            # than ending it.
            rows += [textwrap.indent(description, "    "), ""]
    return rows


def variants_of(field: str, model: str, defs: dict) -> list[tuple[str, str]]:
    """`(tag, definition)` for each variant, by reading the union rather than a list.

    The tag comes first because it is what goes in the file - `name: iou`, not `IouLoss` -
    and the discriminator's mapping is where the two are tied together, so neither is
    transcribed here.
    """
    node = _unwrap(defs[model]["properties"][field])
    if node.get("type") == "array":
        node = _unwrap(node.get("items", {}))
    mapping = node.get("discriminator", {}).get("mapping")
    if not mapping:
        name = _ref_name(node)
        mapping = defs.get(name, {}).get("discriminator", {}).get("mapping", {}) if name else {}
    return [(tag, ref.rsplit("/", 1)[-1]) for tag, ref in mapping.items()]


def _clean_one_line(text: str) -> str:
    """A block's own description, down to its first paragraph: the rest belongs to its fields."""
    first = (text or "").strip().split("\n\n")[0]
    return " ".join(first.split())


def nested_blocks(definition: dict, defs: dict) -> list[tuple[str, str]]:
    """Fields of this block that are themselves blocks with fields of their own.

    `split:` and `augmentation:` are reachable from nowhere else on the page, and a reference
    that omits them is the same shape of wrong as a coverage check that counts - it looks
    complete. Found by asking the schema rather than by listing them here, so a block added
    later appears without this file being touched.
    """
    found = []
    for name, node in definition.get("properties", {}).items():
        node = _unwrap(node)
        if node.get("type") == "array":
            node = _unwrap(node.get("items", {}))
        ref = _ref_name(node)
        if ref and defs.get(ref, {}).get("properties"):
            found.append((name, ref))
    return found


def main() -> None:
    schema = spec_schema()
    defs = schema["$defs"]

    SCHEMA_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_OUT.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")

    example = (ROOT / "examples" / "data_science_bowl.yaml").read_text().rstrip()

    out: list[str] = [
        "---",
        'title: "The configuration file"',
        'description: "Every field a specification accepts, generated from the schema."',
        "---",
        "",
        "<!-- GENERATED by tools/build_config_reference.py from spec_schema(). Do not edit. -->",
        "",
        "A configuration is one YAML file describing a whole run: where the data is, and one",
        "or more models to train on it. **It is the same file in both packages** - R reads it",
        "with `platypus_spec(yaml = ...)` and Python with `from_yaml()`, and both hand it to",
        "the same pydantic models, which is where every description on this page comes from.",
        "",
        "```yaml",
        example,
        "```",
        "",
        "Validation happens before anything is read from disk and before torch is imported, so",
        "a mistake here costs a second rather than an epoch. Every refusal names the field.",
        "",
        "## Schema for your editor",
        "",
        "The same definitions are published as JSON Schema, so an editor can complete and check",
        "a configuration as you type it. In VS Code with the YAML extension, or any editor using",
        "`yaml-language-server`, put this on the first line:",
        "",
        "```yaml",
        f"# yaml-language-server: $schema={schema['$id']}",
        "```",
        "",
        "## `task`, which decides the rest",
        "",
        "`task` is required, and it is the only field that is: it chooses which shape the rest",
        "of the file has. Give it wrong and the refusal names the tag rather than reporting",
        "every field of the other task as unexpected.",
        "",
        "| `task` | what it describes |",
        "|---|---|",
    ]
    # Segmentation first: the mapping's order is the order the union was declared in, which is
    # not a statement about what a reader meets first.
    tasks = sorted(
        schema["discriminator"]["mapping"].items(),
        key=lambda kv: kv[0] != "semantic_segmentation",
    )
    for tag, ref in tasks:
        spec_name = ref.rsplit("/", 1)[-1]
        out.append(f"| `{tag}` | [{spec_name}](#{_task_anchor(tag)}) |")
    out.append("")

    for tag, ref in tasks:
        spec_name = ref.rsplit("/", 1)[-1]
        definition = defs[spec_name]
        short = "segmentation" if tag == "semantic_segmentation" else "detection"
        out += [
            f"## `task: {tag}` {{#{_task_anchor(tag)}}}",
            "",
            f"The {spec_name} shape. Its top level:",
            "",
            *field_table(definition, defs),
            "",
        ]
        # The headings name the task, because both tasks have a `data:` and a `models:`, and
        # two headings with one name leave one of them an anchor nobody can guess.
        for field in ("data", "models"):
            node = _unwrap(definition["properties"].get(field, {}))
            if node.get("type") == "array":
                node = _unwrap(node.get("items", {}))
            block = _ref_name(node)
            if not block:
                continue
            out += [
                f"### `{field}:` for {short}",
                "",
                *field_table(defs[block], defs),
                "",
            ]
            for nested_field, nested_block in nested_blocks(defs[block], defs):
                out += [
                    f"#### `{nested_field}:` inside {short} `{field}:`",
                    "",
                    *field_table(defs[nested_block], defs),
                    "",
                ]

    out += [
        "## Components",
        "",
        "Each of these is named by its `name:` and carries its own fields.",
        "",
    ]
    for title, field, model in FAMILIES:
        out += [f"### {title}", ""]
        for tag, variant in variants_of(field, model, defs):
            definition = defs[variant]
            summary = _clean_one_line(definition.get("description", ""))
            # An explicit id, because `dice`, `iou` and `tversky` are each both a loss and a
            # metric: left to Quarto one of each pair becomes `#dice-1`, an anchor nobody can
            # guess and nothing can link to on purpose. Same defect as two `data:` headings.
            out += [f"#### `{tag}` {{#{field}-{tag}}}", ""]
            if summary:
                out += [summary, ""]
            out += [*field_table(definition, defs), ""]

    OUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} lines, {len(defs)} definitions")
    print(f"wrote {SCHEMA_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
