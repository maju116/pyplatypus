"""Splitting a dataset into train, validation and test - by patient, not by slice.

The single most common way to get a good-looking segmentation result that means nothing:
put slices from one patient into training and validation at the same time. The model sees
the neighbouring slice of the same anatomy in training, so validation measures memory
rather than generalisation, and Dice comes out several points too high with nothing in the
output to say so. Radiology datasets invite it, because they arrive as one flat pile of
slices whose filenames are the only clue about who they came from.

So splitting here is group-aware from the start. `group_by` is a regular expression read
against the sample key; everything sharing a group lands in exactly one split. A group is
usually a patient, and may be a study, a scanner or a site - whatever the unit is that
must not straddle the split.

Two deliberate refusals:

* a `group_by` that matches nothing is an error, never a fallback to one-group-per-sample.
  A silent fallback is exactly the inflated-score bug, reintroduced by a typo.
* the result is written as three CSV files listing paths, not as three copies of the data.
  Medical datasets are large and often read-only, and a split that copies gigabytes is a
  split nobody runs twice.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random

from pyplatypus.data.paths import Sample, discover_samples
from pyplatypus.errors import ConfigError
from pyplatypus.spec.common import DataMode

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class Split:
    """Which samples went where, and which group each one belongs to."""

    train: tuple[Sample, ...]
    validation: tuple[Sample, ...]
    test: tuple[Sample, ...]
    groups: Mapping[str, str]

    def __getitem__(self, name: str) -> tuple[Sample, ...]:
        if name not in SPLIT_NAMES:
            raise KeyError(f"no split called '{name}'; there are: {', '.join(SPLIT_NAMES)}")
        return getattr(self, name)

    @property
    def counts(self) -> dict[str, int]:
        return {name: len(self[name]) for name in SPLIT_NAMES}

    @property
    def group_counts(self) -> dict[str, int]:
        return {name: len({self.groups[s.key] for s in self[name]}) for name in SPLIT_NAMES}

    def groups_of(self, name: str) -> set[str]:
        return {self.groups[s.key] for s in self[name]}


def group_of(key: str, pattern: str | None) -> str:
    """The group a sample belongs to, from its key.

    Without a pattern every sample is its own group, which is the right answer when one
    sample is one patient. With one, the first capture group is the name if the expression
    has one, otherwise the whole match.
    """
    if pattern is None:
        return key
    found = re.search(pattern, key)
    if found is None:
        raise ConfigError(
            f"group_by '{pattern}' does not match the sample key '{key}'. Every sample "
            "must belong to a group; a pattern that matches nothing would quietly make "
            "each sample its own patient, which is the mistake this argument exists to "
            "prevent."
        )
    return found.group(1) if found.groups() else found.group(0)


def _fractions(fractions: Sequence[float] | Mapping[str, float]) -> dict[str, float]:
    if isinstance(fractions, Mapping):
        unknown = set(fractions) - set(SPLIT_NAMES)
        if unknown:
            raise ConfigError(
                f"unknown split(s) {', '.join(sorted(unknown))}; "
                f"there are: {', '.join(SPLIT_NAMES)}"
            )
        shares = {name: float(fractions.get(name, 0.0)) for name in SPLIT_NAMES}
    else:
        if len(fractions) not in (2, 3):
            raise ConfigError(
                f"give two or three fractions (train, validation[, test]), got {len(fractions)}"
            )
        values = [float(v) for v in fractions] + [0.0]
        shares = dict(zip(SPLIT_NAMES, values[:3], strict=True))

    if any(v < 0 for v in shares.values()):
        raise ConfigError(f"fractions cannot be negative: {shares}")
    total = sum(shares.values())
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"fractions must add up to 1, these add up to {total:g}: {shares}")
    if shares["train"] <= 0 or shares["validation"] <= 0:
        raise ConfigError(
            "train and validation must both get a positive share - a model with no "
            "validation set cannot be told whether it learned anything"
        )
    return shares


def split_samples(
    samples: Sequence[Sample],
    *,
    fractions: Sequence[float] | Mapping[str, float] = (0.7, 0.15, 0.15),
    group_by: str | None = None,
    seed: int = 0,
) -> Split:
    r"""Divide samples between the splits, keeping every group whole.

    Deterministic: the same samples, fractions, pattern and seed give the same split on
    any machine. That is not a nicety - a split that moves between runs makes two results
    incomparable, and the reason for a difference becomes impossible to find.

    Groups are shuffled, then handed out one at a time to whichever split is furthest
    below its target share *in samples*. Assigning by group count instead would drift
    badly whenever groups differ in size, which for patients they always do.

    Args:
        samples: What was discovered. Nothing is read from disk here - this partitions a
            list, so a specification with a `split` block gains no side effect on disk.
        fractions: Two or three, summing to one: train, validation, and optionally test.
            A mapping names them instead, for a split that is not in that order.
        group_by: A regular expression read against each sample's key; everything sharing
            a group lands in exactly one split. `None` means every sample is its own
            group, which is right only when the samples are genuinely independent - slices
            of one patient are not.
        seed: Which shuffle. The split is otherwise deterministic.

    Returns:
        A `Split`, with `train`, `validation` and `test` tuples and the group each sample
        was assigned to.

    Raises:
        SplitError: If the fractions are not two or three, if `group_by` matches no sample
                key - a pattern that matches nothing is a mistake, not an empty result - or if
                there are fewer groups than splits to fill.

    >>> from pyplatypus.data.paths import Sample
    >>> import pathlib
    >>> samples = [
    ...     Sample(key=f"patient{p:02d}_slice{s}", images=(pathlib.Path("x.png"),))
    ...     for p in range(1, 5)
    ...     for s in range(3)
    ... ]
    >>> len(samples)
    12
    >>> split = split_samples(samples, fractions=(0.5, 0.5), group_by=r"^(patient\d+)_")
    >>> len(split.train) + len(split.validation)
    12

    A patient is never on both sides, which is the whole point:

    >>> def patients(part):
    ...     return {key.split("_")[0] for key in (sample.key for sample in part)}
    >>> patients(split.train) & patients(split.validation)
    set()

    Without `group_by`, every slice is its own group and the same patient lands in both -
    which makes validation measure memory rather than generalisation:

    >>> loose = split_samples(samples, fractions=(0.5, 0.5))
    >>> bool(patients(loose.train) & patients(loose.validation))
    True
    """
    if not samples:
        raise ConfigError("there are no samples to split")

    shares = _fractions(fractions)
    groups: dict[str, list[Sample]] = {}
    membership: dict[str, str] = {}
    for sample in samples:
        name = group_of(sample.key, group_by)
        groups.setdefault(name, []).append(sample)
        membership[sample.key] = name

    wanted = [name for name in SPLIT_NAMES if shares[name] > 0]
    if len(groups) < len(wanted):
        raise ConfigError(
            f"{len(groups)} group(s) cannot fill {len(wanted)} splits "
            f"({', '.join(wanted)}). With one group per patient, a split needs at least "
            "as many patients as it has parts - or pass a coarser group_by, or ask for "
            "fewer splits."
        )

    order = sorted(groups)
    Random(seed).shuffle(order)

    total = len(samples)
    targets = {name: shares[name] * total for name in wanted}
    assigned: dict[str, str] = {}
    sizes = dict.fromkeys(wanted, 0)

    for name in order:
        # Largest deficit first; ties break on the declared order, so train wins over
        # validation and validation over test. Deterministic either way.
        chosen = max(wanted, key=lambda s: (targets[s] - sizes[s], -wanted.index(s)))
        assigned[name] = chosen
        sizes[chosen] += len(groups[name])

    _fill_empty_splits(assigned, groups, wanted)

    held: dict[str, list[Sample]] = {name: [] for name in SPLIT_NAMES}
    for name in order:
        held[assigned[name]].extend(groups[name])

    return Split(
        train=tuple(held["train"]),
        validation=tuple(held["validation"]),
        test=tuple(held["test"]),
        groups=membership,
    )


def _iteratively_stratified(
    samples: Sequence[Sample],
    *,
    shares: Mapping[str, float],
    group_by: str | None,
    seed: int,
) -> Split:
    """The split for a label that is a **set**: one pass, rarest class first.

    **Strata do not exist here, which is why this is a different algorithm rather than the
    same one with a different key.** `stratified_split` divides the samples by their label
    and splits each part; a sample carrying three findings belongs to three of those parts
    and would be handed out three times. Using the *combination* as the key avoids that and
    buys a guarantee nobody asked for - every observed combination in every part - at a
    price that makes it useless: combinations are as rare as the product of their
    prevalences, so at four independent findings a 4000-image set has combinations with one
    or two members and the split refuses on them.

    So this is iterative stratification (Sechidis, Tsoumakas and Vlahavas, 2011), which
    assigns one unit at a time: take the class with the fewest units left to place, hand
    each of its units to whichever split is furthest below its target **for that class**,
    and repeat. The units with none of the classes at all - the commonest row in any
    dataset at these prevalences - are placed last, on size alone.

    **The unit is the group rather than the sample**, which is what keeps `group_by`'s
    guarantee: a patient's images are placed together and the group's own class counts are
    what the deficit is read against. The single-label path cannot do that - it refuses a
    group spanning two labels, because its strata would hand that group out twice - and
    here a group spanning several findings is the ordinary case, so refusing it would
    refuse nearly every real dataset.

    Measured over 400 draws of four independent findings at 10%, 5%, 3% and 2% - a chest
    film's profile - at 70/15/15, as **how often some part came out holding only one side
    of some class**, which is the state every per-class metric here refuses to average
    over:

        images   random   iterative   iterative refuses   combination refuses
           200   96.75%       0.00%              78.50%                91.75%
           500   51.25%       0.00%               4.50%                96.25%
          1000   12.25%       0.00%               0.00%                97.25%
          4000    0.00%       0.00%               0.00%                94.00%

    The closed form agrees with the random column well enough to say the simulation is
    measuring what it claims: a part of share `f` holds no positive of a class present `r`
    times with probability about `(1-f)^r`, which for the rarest class at 500 images is
    0.1969 per part, and the column is the union over four classes and two small parts.

    **The 0.00% needs its own control, because it was 2.04% on the first reading and the
    cause was not the algorithm.** At 200 images the 2% class is simply absent from 1.5% of
    the draws, and a split cannot notice that: it sees the labels it was handed and nothing
    says a fifth class was expected. `discover_labelled` is where that is refused, by name,
    because the reader is what holds `classes`. Counting those against this function
    reported the reader's job as this algorithm's failure - with them separated the
    iterative column is 0.00% at every size.

    What it costs is nothing anybody can see: the worst drift from the requested fractions
    over those 400 draws is **0.0000** at every size, because the unit is a single image
    and the size deficit is the second key.

    And the refusal column is the honest half of the same measurement. 2% of 200 images is
    four positives, and 15% of four is 0.6, so no assignment can give the smallest part
    one - the refusal says that in those numbers rather than returning a split that dies at
    the end of the first epoch.

    **What the rarest-first order buys is that refusal column and not the correctness**,
    which is worth writing down because reversing it survived every test written for this
    function first. The post-condition below is what makes a returned split correct, so
    the order cannot move the *bad* column at all - it moves how often a split that exists
    is found. Measured over 200 draws with the order reversed, on four independent findings
    and on a rare finding occurring only inside a common one:

        independent      500 images   2.50% refused   ->   3.00% reversed
        nested in one    500 images   3.00% refused   ->  35.50% reversed
        nested in one   1000 images   0.00% refused   ->  12.00% reversed

    The nested shape is the one that separates them, and it is the clinically ordinary one:
    an effusion inside a film already marked abnormal. Placing the common class first
    spends the deficits on groups that carry the rare one too, and then the rare class has
    to fit in what is left.
    """
    groups: dict[str, list[Sample]] = {}
    membership: dict[str, str] = {}
    for sample in samples:
        name = group_of(sample.key, group_by)
        groups.setdefault(name, []).append(sample)
        membership[sample.key] = name

    wanted = [name for name in SPLIT_NAMES if shares[name] > 0]
    if len(groups) < len(wanted):
        raise ConfigError(
            f"{len(groups)} group(s) cannot fill {len(wanted)} splits "
            f"({', '.join(wanted)}). With one group per patient, a split needs at least "
            "as many patients as it has parts - or pass a coarser group_by, or ask for "
            "fewer splits."
        )

    classes = sorted({name for sample in samples for name in (sample.labels or ())})
    carried = {
        group: Counter(name for sample in members for name in (sample.labels or ()))
        for group, members in groups.items()
    }
    found = Counter()
    for counted in carried.values():
        found.update(counted)

    want = {(klass, part): shares[part] * found[klass] for klass in classes for part in wanted}
    want_size = {part: shares[part] * len(samples) for part in wanted}
    held = dict.fromkeys(want, 0.0)
    sizes = dict.fromkeys(wanted, 0)

    order = sorted(groups)
    Random(seed).shuffle(order)
    left = set(order)
    assigned: dict[str, str] = {}

    def place(group: str, klass: str | None) -> None:
        # Ties break on the declared order, so train wins over validation and validation
        # over test - the same rule `split_samples` uses, and the reason two runs of this
        # with one seed are the same split.
        if klass is None:
            chosen = max(
                wanted, key=lambda part: (want_size[part] - sizes[part], -wanted.index(part))
            )
        else:
            chosen = max(
                wanted,
                key=lambda part: (
                    want[(klass, part)] - held[(klass, part)],
                    want_size[part] - sizes[part],
                    -wanted.index(part),
                ),
            )
        assigned[group] = chosen
        sizes[chosen] += len(groups[group])
        for name, count in carried[group].items():
            held[(name, chosen)] += count
        left.discard(group)

    while left:
        remaining = Counter()
        for group in left:
            remaining.update(carried[group])
        candidates = [klass for klass in classes if remaining[klass] > 0]
        if candidates:
            # The rarest first, because it is the one with no room for a bad decision: by
            # the time a common class is placed the deficits are what the rare ones left
            # behind, and that is the trade this order makes on purpose.
            klass = min(candidates, key=lambda name: (remaining[name], name))
            batch = [group for group in order if group in left and carried[group][klass] > 0]
        else:
            klass = None
            batch = [group for group in order if group in left]
        for group in batch:
            place(group, klass)

    _fill_empty_splits(assigned, groups, wanted)

    places: dict[str, list[Sample]] = {name: [] for name in SPLIT_NAMES}
    for group in order:
        places[assigned[group]].extend(groups[group])

    # The post-condition, checked rather than argued for: every class needs a positive and
    # a negative in every part, because that is what each per-class metric averages over
    # and a part holding one side of a class kills the run at the end of its first epoch.
    # The arithmetic is why it can fail - a class with 3 positives over 70/15/15 has 0.45
    # of an image due to the smallest part, so no assignment puts one there - and the
    # smallest share is what makes that readable as a number rather than as bad luck.
    smallest = min(shares[part] for part in wanted)
    for klass in classes:
        for part in wanted:
            positive = sum(1 for sample in places[part] if klass in (sample.labels or ()))
            if positive and positive < len(places[part]):
                continue
            side = "no image carrying" if not positive else "nothing but images carrying"
            raise ConfigError(
                f"the '{part}' split came out with {side} '{klass}', which every per-class "
                f"metric for a set of labels refuses to average over. It has {found[klass]} "
                f"of {len(samples)} images and the smallest part asked for is "
                f"{smallest:.0%}, so a class needs about {int(1 / smallest) + 1} images on "
                f"each side of it before a split can give both to every part - take the "
                f"class out of `classes`, ask for fewer parts, or find more data."
            )

    return Split(
        train=tuple(places["train"]),
        validation=tuple(places["validation"]),
        test=tuple(places["test"]),
        groups=membership,
    )


def stratified_split(
    samples: Sequence[Sample],
    *,
    fractions: Sequence[float] | Mapping[str, float] = (0.7, 0.15, 0.15),
    group_by: str | None = None,
    seed: int = 0,
) -> Split:
    """`split_samples` once per label, so no split comes out missing a class.

    **A label that is a set is a different algorithm and this dispatches to it**, rather
    than being two functions a caller has to choose between: the question is the same one -
    does every part come out able to score every class - and the answer for a set of labels
    cannot be strata, because a film carrying three findings belongs to three of them.
    `_iteratively_stratified` has that half and the measurement for it. Which it is follows
    from the samples: `labels` set means a set, and a dataset labelled both ways is refused.

    **This is not a refinement, it is what a classification split needs to be.** Measured
    over 400 seeds at 70/15/15, how often some part held none of the rare class:

        images   prevalence   random   stratified
           100         3%     91.25%        0.00%
           200         3%     63.75%        0.00%
           500         3%     19.75%        0.00%
          1000         3%      0.75%        0.00%
          2000         1%      6.75%        0.00%

    and the closed form agrees with the simulation - a part of share `f` holds none of `r`
    rare items with probability about `(1-f)^r`, so with three rare items the 15% parts miss
    them 0.85^3 = **0.6141** of the time each and some part misses them 0.9055 of the time
    against the **0.9125** measured above, which is half a standard error at 400 seeds.

    **That sentence quoted 0.5120 against a simulation of 51.18% until `pyplatypus#329`**,
    and neither belongs to this table: 0.5120 is exactly `0.8^3`, a part of share 0.2, and
    the nearest figure in this file is the **51.25%** of the multi-label table 180 lines
    above - a different algorithm answering a different question. §4's number measured in one
    setting and quoted in another. What it costs is nothing anybody can see: the worst drift from the requested
    fractions over 200 seeds is 0.0100 at 100 images, which is one image, and 0.0000 at 500.

    The alternative is not a slightly worse split. Every per-class metric this task offers -
    `balanced_accuracy`, `f1`, `auc` - refuses to average over a class the pass never
    contained, so a split with an empty class fails at the end of the first epoch, after
    the epoch is paid for.

    **A wrapper rather than a flag inside `split_samples`**, and deliberately: the
    assignment it does - shuffle the groups, hand each to whichever split is furthest below
    its target, then make sure none came out empty - is exactly what each stratum needs, so
    this calls it rather than reaching into it. One seed for every stratum, measured: the
    parts come out balanced at every seed without deriving one per label.

    Args:
        samples: What was discovered. Every one must carry a `label`.
        fractions: As `split_samples` takes them.
        group_by: As `split_samples` takes it, and **refused when any group holds more than
            one label** - see below.
        seed: Which shuffle, the same for every stratum.

    Returns:
        A `Split`, with the groups every stratum assigned.

    Raises:
        ConfigError: If a sample carries no label, if `group_by` matches a group holding
            more than one label, or - from `split_samples`, per stratum - if a class has
            fewer examples than there are splits to fill.
    """
    if not samples:
        raise ConfigError("there are no samples to split")

    sets = [sample for sample in samples if sample.labels is not None]
    if sets:
        if len(sets) != len(samples):
            named = [sample.key for sample in samples if sample.labels is None]
            shown = ", ".join(named[:5]) + (" ..." if len(named) > 5 else "")
            raise ConfigError(
                f"{len(named)} of {len(samples)} samples carry one label where the rest "
                f"carry a set of them: {shown}. The two need different algorithms - strata "
                f"for one, one pass rarest-class-first for a set - so there is no split "
                f"that is correct for both halves."
            )
        # `_fractions` is called here rather than above so that the single-label path's
        # refusals arrive in the order they always have: it is `split_samples` that
        # validates them there, one stratum in.
        return _iteratively_stratified(
            samples, shares=_fractions(fractions), group_by=group_by, seed=seed
        )

    unlabelled = [sample.key for sample in samples if sample.label is None]
    if unlabelled:
        shown = ", ".join(unlabelled[:5]) + (" ..." if len(unlabelled) > 5 else "")
        raise ConfigError(
            f"{len(unlabelled)} sample(s) carry no label, so there is nothing to stratify "
            f"by: {shown}. A stratified split is for a task whose samples are labelled with "
            f"a name; `split_samples` is the one that does not need one."
        )

    if group_by is not None:
        # Stratifying cuts the samples by label and then splits each part, so a group
        # spanning two labels is handed out twice - once in each stratum - and lands on
        # both sides. That is the leak `group_by` exists to prevent, so the combination is
        # refused rather than silently preferring one guarantee over the other. Measured on
        # six patients with two of them holding both labels: `patient06` appeared in train
        # and in validation, where the same data split without stratifying leaked nothing.
        labels_of: dict[str, set[str]] = {}
        for sample in samples:
            labels_of.setdefault(group_of(sample.key, group_by), set()).add(sample.label)
        mixed = sorted(name for name, labels in labels_of.items() if len(labels) > 1)
        if mixed:
            shown = ", ".join(mixed[:5]) + (" ..." if len(mixed) > 5 else "")
            raise ConfigError(
                f"{len(mixed)} of {len(labels_of)} groups hold more than one label, and a "
                f"stratified split cannot keep those groups whole: {shown}. Stratifying "
                f"divides the samples by label first, so such a group is handed out once "
                f"per label it contains and ends up on both sides - which is exactly what "
                f"`group_by` is for. Either drop `group_by`, or use a coarser pattern whose "
                f"groups are each of one class."
            )

    strata: dict[str, list[Sample]] = {}
    for sample in samples:
        strata.setdefault(sample.label, []).append(sample)

    held: dict[str, list[Sample]] = {name: [] for name in SPLIT_NAMES}
    membership: dict[str, str] = {}
    for label, members in sorted(strata.items()):
        try:
            part = split_samples(members, fractions=fractions, group_by=group_by, seed=seed)
        except ConfigError as refusal:
            # The refusal from inside a stratum is about that class, and `split_samples`
            # cannot know that - its own message talks about groups and patients, which is
            # the right message for the caller it was written for and the wrong one here.
            raise ConfigError(
                f"class '{label}' cannot be split: {refusal}. A stratified split gives "
                f"every class to every part, so a class needs at least as many examples as "
                f"there are parts - remove it from `classes` if it is not really in this "
                f"dataset, or ask for fewer splits."
            ) from refusal
        for name in SPLIT_NAMES:
            held[name].extend(getattr(part, name))
        membership.update(part.groups)

    return Split(
        train=tuple(held["train"]),
        validation=tuple(held["validation"]),
        test=tuple(held["test"]),
        groups=membership,
    )


def _fill_empty_splits(
    assigned: dict[str, str], groups: Mapping[str, list[Sample]], wanted: Sequence[str]
) -> None:
    """Make sure every requested split got something, by moving one group if it did not.

    Chasing the target *sizes* can leave a split empty when the groups are lopsided: one
    patient with 100 slices and two with one each fills train and starves test, however the
    shuffle came out. An empty validation or test set is useless, and refusing the whole
    split over an unlucky shuffle is worse than being slightly off the requested fractions,
    so the smallest group is moved out of whichever split holds the most.
    """
    for name in wanted:
        if any(target == name for target in assigned.values()):
            continue
        counts: dict[str, list[str]] = {}
        for group, target in assigned.items():
            counts.setdefault(target, []).append(group)
        donor = max(counts, key=lambda s: (len(counts[s]), -wanted.index(s)))
        if len(counts[donor]) < 2:
            raise ConfigError(
                f"the '{name}' split came out empty and no other split has a group to "
                "spare. Ask for fewer splits, or use a coarser group_by."
            )
        moved = min(counts[donor], key=lambda g: (len(groups[g]), g))
        assigned[moved] = name


MASK_PATHS = "mask paths"
A_CLASS_NAME = "a class name"
A_SET_OF_LABELS = "a set of labels"


def _labelling_of(samples: Sequence[Sample]) -> str:
    """Which of the three kinds of label these samples carry, refusing a mixture.

    Read off the samples rather than taken as a flag, for `stratified_split`'s reason one
    layer over: a caller holding a `Split` holds the answer already, and a flag beside it is
    a second copy of it that can disagree.

    **The first that applies wins** - `labels` is the set, `label` the name, masks are what
    is left - and no walk in this package produces a sample with two of them, since each
    reader fills exactly one. A hand-built `Sample` carrying both a `label` and `masks` is
    written as the label; refusing that here would be a guard on a state nothing can reach
    through its own routes, and the place to refuse a malformed sample is `Sample`.

    Raises:
        ConfigError: If the samples do not agree, because one file has one header.
    """
    kinds: dict[str, list[str]] = {}
    for sample in samples:
        if sample.labels is not None:
            kinds.setdefault(A_SET_OF_LABELS, []).append(sample.key)
        elif sample.label is not None:
            kinds.setdefault(A_CLASS_NAME, []).append(sample.key)
        else:
            kinds.setdefault(MASK_PATHS, []).append(sample.key)
    if len(kinds) > 1:
        parts = []
        for kind, keys in sorted(kinds.items()):
            shown = ", ".join(keys[:5]) + (" ..." if len(keys) > 5 else "")
            parts.append(f"{len(keys)} carry {kind} ({shown})")
        raise ConfigError(
            f"these samples do not agree on what a label is: {'; '.join(parts)}. One CSV has "
            f"one header, so there is no file that is correct for all of them."
        )
    return next(iter(kinds))


def write_splits(
    split: Split,
    out_dir: str | Path,
    *,
    column_sep: str = ";",
    relative: bool = True,
    label_column: str | None = None,
    classes: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write one CSV per split, in the layout `config_file` mode reads.

    Paths are written relative to the CSV when they can be, so the dataset and its split
    can be moved or mounted elsewhere together. `key` and `group` columns come along for
    provenance: the reader ignores them, a person reading the file does not, and they are
    what lets anyone check afterwards that no patient straddles the split.

    **What the label column holds is decided by the samples**, not by the caller, and that
    is `#316`: this wrote `sample.masks` whatever the task, so a classification split came
    out with an empty column and the dataset's labelling was gone. The refusal arrived one
    run later and at the reader - `no label name` on every row - which is a sentence about
    the data rather than about the file that had just been written. The three shapes:

        mask paths        one `masks` column of paths, which is every mask task
        a class name      one `label` column of names, which is `classification`
        a set of labels   one column per class, `1` or `0`, which is `labels_from`

    **A set of labels needs `classes`**, and it is refused without them rather than taken
    from the samples. The union of what a split happens to carry is not the class list: a
    class no image in this part has would simply have no column, and the file would then
    describe a narrower problem than the one being studied - which is the single thing
    `classes` exists to stop. It is the same argument, and the same parameter name, as
    `save_class_masks` and `save_class_volumes`.

    Args:
        split: What to write, from `split_samples` or `stratified_split`.
        out_dir: Where the three CSVs go. Created if it does not exist.
        column_sep: What separates several paths inside one cell.
        relative: Write paths relative to the CSV where that is possible.
        label_column: Override the label column's name. `None` takes the name from what the
            samples carry - `masks` for paths and `label` for a class name, which is what
            `ClassificationData.label_column` reads. Ignored for a set of labels, where the
            class names *are* the column names.
        classes: The classes, in class order, and required for a set of labels. The reader
            pairs its columns to `classes` by position, so this is also what fixes each
            class's index.

    Returns:
        The path written for each non-empty split, keyed by split name.

    Raises:
        ConfigError: If the samples disagree about what a label is, if a set of labels is
            written without `classes`, or if a sample carries a class `classes` does not
            name.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    everything = [sample for name in SPLIT_NAMES for sample in split[name]]
    if not everything:
        # Nothing to write and nothing to decide. Reachable by constructing a `Split`
        # directly; every function in this module that builds one refuses an empty result
        # before it gets here.
        return {}

    labelling = _labelling_of(everything)
    if labelling == A_SET_OF_LABELS:
        if classes is None:
            raise ConfigError(
                "these samples carry a set of labels, which is written as one column per "
                "class, so `classes` is needed to say which columns and in what order. The "
                "classes the samples happen to carry are not that list: a class none of "
                "them has would have no column at all."
            )
        named = set(classes)
        stray = sorted({k for s in everything for k in (s.labels or ()) if k not in named})
        if stray:
            raise ConfigError(
                f"these samples carry {stray}, which `classes` does not name. One column "
                f"per class is the layout, so a finding with no column would be dropped "
                f"from the file with nothing saying so. `classes` is {list(classes)}."
            )
        header = ["key", "group", "images", *classes]
    else:
        default = "label" if labelling == A_CLASS_NAME else "masks"
        header = ["key", "group", "images", label_column or default]

    written: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        samples = split[name]
        if not samples:
            continue
        path = out / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for sample in samples:
                if labelling == A_SET_OF_LABELS:
                    carried = sample.labels or ()
                    label_cells = ["1" if klass in carried else "0" for klass in classes or ()]
                elif labelling == A_CLASS_NAME:
                    label_cells = [sample.label or ""]
                else:
                    label_cells = [
                        column_sep.join(_as_text(p, out, relative) for p in sample.masks)
                    ]
                writer.writerow(
                    [
                        sample.key,
                        split.groups[sample.key],
                        column_sep.join(_as_text(p, out, relative) for p in sample.images),
                        *label_cells,
                    ]
                )
        written[name] = path
    return written


def _as_text(path: Path, base: Path, relative: bool) -> str:
    if not relative:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        # Not under the output directory. An absolute path still works from anywhere,
        # which is better than a fragile chain of '..' segments.
        return str(path.resolve())


def split_dataset(
    root: str | Path,
    out_dir: str | Path,
    *,
    mode: DataMode | str = DataMode.NESTED_DIRS,
    subdirs: tuple[str, str] = ("images", "masks"),
    column_sep: str = ";",
    fractions: Sequence[float] | Mapping[str, float] = (0.7, 0.15, 0.15),
    group_by: str | None = None,
    seed: int = 0,
    strict: bool = True,
    relative: bool = True,
    label_column: str = "masks",
) -> dict[str, object]:
    r"""Split one folder of data into three CSVs a specification can point at.

    The whole point of the function: a researcher has one directory and needs
    `train_path`, `validation_path` and `test_path`. Returns the paths it wrote plus the
    counts, so the caller can see what happened without opening the files.

    `label_column` names the second column, and it has to match what the specification
    will look for - `masks` for segmentation, `annotations` for detection. Not derived
    from `subdirs`, although that would read well: someone whose directories are called
    `img` and `lbl` has always got a `masks` column out of this, and deriving it would
    quietly write a file their existing configuration could no longer read.

    The other way to divide one folder is the specification's own `split` block, which
    partitions the samples in memory and writes nothing. Use this one when the CSVs are
    the point - to keep, to hand to a colleague, or to cite in a paper.

    Args:
        root: The folder holding the data.
        out_dir: Where the three CSVs go.
        mode: `nested_dirs` for one directory per sample, `config_file` for a CSV.
        subdirs: For `nested_dirs`, the image and label subdirectories.
        column_sep: What separates several paths **inside one cell** - a sample with one
            mask file per object, or one file per channel. Not the CSV's own delimiter,
            which is a comma: that is the distinction this argument's name does not make.
        fractions: Two or three, as for `split_samples`.
        group_by: A pattern picking the group out of each sample's key - a patient,
            usually. As for `split_samples`, and for the same reason.
        seed: Which shuffle.
        strict: Whether a sample that cannot be read is an error rather than a skip.
        relative: Write paths relative to `out_dir`, so the CSVs survive being moved
            together with the data.
        label_column: The name of the second column.

    Returns:
        `paths` - the CSV written for each split that got any; `samples` and `groups` -
        how many of each went where, test included and zero when there are two fractions;
        and `skipped`, which is not zero when `strict` is False and something could not be
        read.

    >>> import pathlib, tempfile
    >>> root = pathlib.Path(tempfile.mkdtemp())
    >>> for patient in range(1, 5):
    ...     for slice_no in range(2):
    ...         case = root / f"patient{patient:02d}_slice{slice_no}"
    ...         (case / "images").mkdir(parents=True)
    ...         (case / "masks").mkdir(parents=True)
    ...         _ = (case / "images" / "scan.png").write_bytes(b"")
    ...         _ = (case / "masks" / "mask.png").write_bytes(b"")
    >>> out = split_dataset(
    ...     root,
    ...     root / "splits",
    ...     fractions=(0.5, 0.5),
    ...     group_by=r"^(patient\d+)_",
    ... )
    >>> sorted(out)
    ['groups', 'paths', 'samples', 'skipped']
    >>> out["samples"]
    {'train': 4, 'validation': 4, 'test': 0}

    Two groups each side, because a patient is never divided:

    >>> out["groups"]
    {'train': 2, 'validation': 2, 'test': 0}

    And the CSVs are what a specification points at. Note the header: the file is
    comma-separated and carries the key and the group it was assigned, while `column_sep`
    separates several paths within a single cell.

    >>> written = pathlib.Path(out["paths"]["train"]).read_text().splitlines()
    >>> written[0]
    'key,group,images,masks'
    >>> len(written) - 1
    4
    """
    found = discover_samples(
        root,
        mode=DataMode(mode),
        subdirs=subdirs,
        column_sep=column_sep,
        strict=strict,
        label_column=label_column,
    )
    split = split_samples(found.samples, fractions=fractions, group_by=group_by, seed=seed)
    paths = write_splits(
        split, out_dir, column_sep=column_sep, relative=relative, label_column=label_column
    )
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "samples": split.counts,
        "groups": split.group_counts,
        "skipped": len(found.skipped),
    }
