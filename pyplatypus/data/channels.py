"""Matching a sample's files to the channels a model expects.

Some datasets keep one channel per file. BraTS ships four MRI sequences per patient - T1,
T1 after contrast, T2 and T2-FLAIR - because different tumour structures are visible in
different sequences; the 38-Cloud satellite set keeps its bands the same way. The model wants
them stacked in a known order, and "known" is the whole difficulty.

**Alphabetical order is not an answer.** Sorted, BraTS gives flair, t1, t1ce, t2 - which is a
perfectly reproducible order that means nothing anatomically. A model trained with FLAIR in
channel one and then used on data where channel one is T1 produces a plausible-looking result
and no error whatsoever. Nothing downstream can detect it: the arrays have the right shape and
the right range.

So the order is stated, as a pattern per channel, and every pattern must match exactly one of
the sample's files. Matching none is a typo or a missing file; matching two means the pattern
is not specific enough - `_t1` matches both `_t1.nii.gz` and `_t1ce.nii.gz`, which is the
mistake this module exists to make loud.
"""

from __future__ import annotations

import re
from pathlib import Path

from pyplatypus.errors import PlatypusError


class ChannelError(PlatypusError):
    kind = "channel_error"


def match_channels(
    paths: tuple[Path, ...] | list,
    patterns: list[str],
    *,
    key: str | None = None,
    what: str = "channel",
) -> tuple[Path, ...]:
    """The sample's files in `what` order, one per pattern.

    Patterns are regular expressions searched against each file's name. `key` names the sample
    in any complaint, because "pattern matched nothing" is not actionable without it.

    `what` is the noun every message uses, because this matcher answers two questions with
    identical rules: `channels_from` names one file per input channel, and `classes_from`
    names one file per class of the target. Every guard below is the same in both - exactly
    one file per pattern, no file matched twice, no file left out - and the only difference
    is whether a mismatch mislabels a measurement or a class. A second copy with the other
    noun is the shape this project keeps paying for, so there is one function and a word.
    """
    where = f" in sample '{key}'" if key else ""
    files = [Path(p) for p in paths]

    chosen: list[Path] = []
    for position, pattern in enumerate(patterns, start=1):
        try:
            matcher = re.compile(pattern)
        except re.error as error:
            raise ChannelError(
                f"{what} {position} pattern '{pattern}' is not a valid regular expression: {error}"
            ) from None

        hits = [path for path in files if matcher.search(path.name)]
        if not hits:
            listed = ", ".join(sorted(path.name for path in files)) or "none"
            raise ChannelError(
                f"{what} {position} pattern '{pattern}' matches no file{where}. Files "
                f"present: {listed}."
            )
        if len(hits) > 1:
            listed = ", ".join(sorted(path.name for path in hits))
            raise ChannelError(
                f"{what} {position} pattern '{pattern}' matches {len(hits)} files{where}: "
                f"{listed}. A pattern has to pick exactly one, or the channel order depends "
                "on which file was listed first - '_t1' matches '_t1.nii.gz' and "
                "'_t1ce.nii.gz' alike. Make it more specific, for example '_t1\\\\.nii'."
            )
        chosen.append(hits[0])

    repeated = {path for path in chosen if chosen.count(path) > 1}
    if repeated:
        listed = ", ".join(sorted(path.name for path in repeated))
        raise ChannelError(
            f"two {what}s matched the same file{where}: {listed}. Every {what} is a "
            "different thing, so one file cannot be two of them."
        )

    unused = [path for path in files if path not in chosen]
    if unused:
        listed = ", ".join(sorted(path.name for path in unused))
        raise ChannelError(
            f"{len(unused)} file(s){where} match no {what} pattern: {listed}. Left out "
            "silently they would be data the model never sees; name them or remove them."
        )
    return tuple(chosen)
