"""Every `` `name()` `` the newest CHANGELOG entry offers a reader has to resolve.

`pyplatypus#440`. A changelog entry saying *"an R user gains `load_weights("fives-unet")`"*
is an instruction telling somebody to go and call something, and there is no such function
in either package. It shipped in `0.8.0a77`, on `main` and inside a tag.

**This file had already written down its own gap and nothing closed it.** `0.8.0a72`'s entry
says, about the identical defect one package over:

    The same false claim was written into R's own `NEWS.md` entry for this pin and was
    caught there by §3F's NEWS-prose rule ... The rule cannot read this file, which is
    why one copy was caught and the other had to be read.

Five releases later the same class recurred here. §4 has the category - a limitation stated
in prose and designed around is a defect with an alibi - and this is it in a changelog
rather than in a docstring.

THE SCOPE IS THE NEWEST ENTRY, which is R's design and its reason transfers unchanged: over
the whole file a changelog legitimately names functions that have been **deleted**, and that
is often why they are mentioned at all. Measured here over the last sixteen entries, four
names resolve in neither package and three of the four are legitimate - `DetectionMetrics.as_rows`,
removed by the very entry that names it; `specification`, which lives in `examples/` and not
in the wheel; and `is_available`, which is torch's. A test over the whole file would be §4's
badly calibrated check in the direction nobody warns about: tight enough to fail on correct
content. Only the entry being written now is anybody's to get right.

**R's names are CITED rather than offered, and that asymmetry is structural rather than a
preference.** R's half of this rule resolves names in both namespaces because R *runs* the
engine and can ask it; the engine cannot run R, and making its suite fetch `NAMESPACE` over
the network would put a flaky dependency in the one place this project's own notes say has
the worst CI-reading record. So the rule here is that `` `name()` `` means an ENGINE name,
and an R function in an engine entry is written in backticks **without** parentheses - which
is the convention this project already uses for a name being cited rather than offered, and
is what lets a correction name a dead function without failing on itself.

The resolver is deliberately generous: module-level names, class attributes and dotted
`Class.method` forms all count, because a changelog legitimately names a method. What it
refuses is a name that resolves NOWHERE, which is the defect that has now occurred twice.
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import pyplatypus as pp

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
OFFERED = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)\(\)`")


def _engine_names() -> set[str]:
    """Every name a reader could reach from the package, methods and dotted forms included."""
    names: set[str] = set()
    modules = [pp]
    skipped = []
    for info in pkgutil.walk_packages(pp.__path__, prefix="pyplatypus."):
        try:
            modules.append(importlib.import_module(info.name))
        except ImportError as absent:  # an optional extra is not installed in every job
            skipped.append(f"{info.name}: {absent}")
    assert len(skipped) < 5, (
        f"too many modules did not import, so this resolver is hollow: {skipped}"
    )
    for module in modules:
        for name in dir(module):
            if name.startswith("_"):
                continue
            names.add(name)
            thing = getattr(module, name, None)
            if inspect.isclass(thing):
                for attribute in dir(thing):
                    if not attribute.startswith("_"):
                        names.add(attribute)
                        names.add(f"{name}.{attribute}")
    return names


def _newest_entry() -> tuple[str, str]:
    text = CHANGELOG.read_text()
    starts = [m.start() for m in re.finditer(r"^## \[", text, re.MULTILINE)]
    assert len(starts) > 1, "no version headings found, so this test is reading the wrong file"
    block = text[starts[0] : starts[1]]
    return block.split("\n", 1)[0].strip(), block


def test_the_resolver_can_see_the_package_at_all():
    """The control. An empty name set reports every entry clean and reads exactly like a
    file with nothing wrong in it - §4's count that comes back zero because the instrument
    is not running."""
    names = _engine_names()
    assert len(names) > 500, f"the resolver found only {len(names)} names"
    for known in ("from_dict", "available_weights", "export_weights"):
        assert known in names, f"{known} is a public name of this package and did not resolve"


def test_the_newest_changelog_entry_was_found_and_offers_something():
    """A entry with no `name()` in it passes by having nothing to check, which is a silent
    instrument rather than a clean bill. This asserts the reading happened; it does not
    require an entry to name anything, because some genuinely do not."""
    heading, block = _newest_entry()
    assert heading.startswith("## ["), heading
    assert len(block) > 200, f"{heading} is too short to be an entry: {len(block)} characters"


def test_every_name_the_newest_changelog_entry_OFFERS_resolves_in_the_engine():
    heading, block = _newest_entry()
    offered = sorted(set(OFFERED.findall(block)))
    names = _engine_names()
    unresolved = [name for name in offered if name not in names]
    assert not unresolved, (
        f"{heading} offers {unresolved} and this package answers to none of them.\n"
        "A name in backticks WITH parentheses is an instruction to go and call something. "
        "If it is an R function, write it without the parentheses - that is this project's "
        "notation for a name being cited rather than offered, and it is what lets an entry "
        "name a function that does not exist here without claiming it does."
    )


@pytest.mark.parametrize("invented", ["load_weights", "platypus_engine_report", "loss_cldice"])
def test_the_rule_would_have_caught_the_names_it_was_written_for(invented):
    """The positive half, on the three real ones rather than on something made up.

    `load_weights` is `0.8.0a77`'s, `platypus_engine_report` is `rplatypus#229`'s - the same
    defect one package over, which is where this rule comes from - and `loss_cldice` is the
    one `0.8.0a72` recorded being caught in R and NOT here, in the paragraph that says this
    file cannot be read by the rule.
    """
    assert invented not in _engine_names()
