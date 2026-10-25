"""The contrast ratios `_brand.yml` writes down, recomputed from the colours it declares.

A contrast ratio is a measured number in prose, and CLAUDE.md §3C says such a number needs
something holding it to the measurement or it should not be written. Nothing held this one:
the table in that file was wrong in three of its four rows from the day it was written -
brown 11.8 against 12.7, terracotta 3.7 against 3.4, tan 2.6 against 2.4 - and nothing
noticed, because every conclusion drawn from it happened to survive the error.

That is the shape this project keeps paying for: an error that changes no decision is an
error nobody looks for. The table decides which colour may carry a link, so the next one
that is wrong might change one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

BRAND = Path(__file__).resolve().parents[1] / "_brand.yml"

#: The four values the WCAG definition fixes, which is this test's own control: a formula
#: that gets these right is the formula, and one that does not cannot judge anything else.
KNOWN = {"#000000": 21.00, "#ffffff": 1.00, "#767676": 4.54, "#595959": 7.00}


def contrast_on_white(colour: str) -> float:
    """The WCAG 2.x ratio of `colour` against white.

    Written out rather than imported so that the test does not inherit the error it exists
    to catch from whatever wrote the table.
    """
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return 1.05 / (luminance + 0.05)


@pytest.mark.parametrize(("colour", "ratio"), sorted(KNOWN.items()))
def test_the_formula_is_the_formula(colour: str, ratio: float):
    """The control, first: these four are fixed by the definition rather than by taste."""
    assert contrast_on_white(colour) == pytest.approx(ratio, abs=0.01)


def _declared() -> tuple[dict[str, str], list[tuple[str, str, float]]]:
    text = BRAND.read_text()
    palette = yaml.safe_load(text)["color"]["palette"]
    # The rows of the CONTRAST table: a name, its hex, and the ratio the file claims.
    #
    # The negative lookahead is not decoration. This file carries a second table of the same
    # shape - the cluster shares the palette was derived from - and without it this regex
    # read `sage #d9dbba 37.0%` as a contrast ratio of 37. An instrument that matches the
    # wrong table answers a question nobody asked, which is the first thing to suspect when
    # a check fails on something that looks right.
    rows = re.findall(r"^#\s+(\w+)\s+(#[0-9a-f]{6})\s+(\d+\.\d)(?!%)\s", text, re.MULTILINE)
    return palette, [(name, colour, float(ratio)) for name, colour, ratio in rows]


def test_every_ratio_the_brand_file_writes_down_is_the_one_it_would_measure():
    """And the colours in the table are the colours in the palette, not near-misses of them."""
    palette, rows = _declared()
    assert rows, "no contrast table found in _brand.yml; the regex is the instrument here"

    for name, colour, claimed in rows:
        assert name in palette, f"the table names {name!r}, which the palette does not define"
        assert palette[name] == colour, (
            f"the table gives {name} as {colour} and the palette as {palette[name]}"
        )
        assert contrast_on_white(colour) == pytest.approx(claimed, abs=0.05), (
            f"{name} {colour} is {contrast_on_white(colour):.2f} against white and the file "
            f"says {claimed}"
        )


def test_the_colour_that_carries_links_clears_the_threshold_it_is_chosen_for():
    """`primary` is what every link is drawn in, so 4.5 is a constraint and not a target."""
    palette, _ = _declared()
    brand = yaml.safe_load(BRAND.read_text())["color"]
    primary = palette[brand["primary"]]
    assert contrast_on_white(primary) >= 4.5, (
        f"`primary` is {brand['primary']} {primary} at {contrast_on_white(primary):.2f} "
        "against white, which is below what body text needs"
    )

    # And the one the file says must NOT carry a link, so that the rule is asserted from
    # both ends rather than only the comfortable one.
    assert contrast_on_white(palette["tan"]) < 4.5, (
        "tan clears 4.5 now, so the sentence saying it must not carry a link is stale"
    )
