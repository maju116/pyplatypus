"""Shuffling a tiled dataset in windows, so a decoded sample serves all of its tiles.

The cache was there from the start - `_load` says "cached because every tile asks again" -
and shuffling over tile indices defeated it: eight samples held against a working set of six
hundred. Measured on FIVES, the loader cost 254 ms a tile shuffled against 49 in order.
"""

from __future__ import annotations

import pytest
import torch

from pyplatypus.training.torch_data import TileShuffle, make_loader


def test_it_is_a_permutation():
    """Every tile once per epoch. A sampler that drops or repeats would train on a
    different dataset from the one asked for, and the score would not say so."""
    sampler = TileShuffle(samples=20, tiles=4, window=3)

    drawn = list(sampler)
    assert len(drawn) == len(sampler) == 80
    assert sorted(drawn) == list(range(80))


def test_a_sample_is_decoded_once_because_its_tiles_share_a_window():
    """The property the whole thing exists for, asserted on the order rather than on a
    timing: within any run of `window * tiles` emitted indices, at most `window` distinct
    samples appear - so a cache of `window` never evicts a sample still being asked for."""
    window, tiles = 4, 16
    sampler = TileShuffle(samples=50, tiles=tiles, window=window)

    drawn = list(sampler)
    for start in range(0, len(drawn), window * tiles):
        block = drawn[start : start + window * tiles]
        assert len({index // tiles for index in block}) <= window


def test_a_batch_still_draws_from_several_samples():
    """What the window does not give up. One sample's tiles emitted together would make a
    batch sixteen views of one retina, which is not the same gradient; this keeps the
    mixing that full shuffling had, inside the window."""
    sampler = TileShuffle(samples=40, tiles=16, window=8)

    drawn = list(sampler)
    batches = [drawn[i : i + 8] for i in range(0, 80, 8)]
    sources = [len({index // 16 for index in batch}) for batch in batches]

    assert min(sources) > 1, f"a batch came from one sample: {sources}"
    assert sum(sources) / len(sources) > 2


def test_the_same_seed_draws_the_same_order():
    torch.manual_seed(7)
    first = list(TileShuffle(samples=12, tiles=4, window=3))
    torch.manual_seed(7)
    again = list(TileShuffle(samples=12, tiles=4, window=3))
    torch.manual_seed(8)
    other = list(TileShuffle(samples=12, tiles=4, window=3))

    assert first == again
    assert first != other, "two seeds that agree would mean `seed` reaches nothing here"


def test_the_window_is_never_zero():
    assert list(TileShuffle(samples=4, tiles=2, window=0))


class _Tiled:
    """The least a `make_loader` call needs: a length, a tile count and a cache size."""

    def __init__(self, samples: int, tiles: int, cache_size: int = 8):
        self.tiles_per_sample = tiles
        self.cache_size = cache_size
        self._n = samples * tiles

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        return torch.zeros(1, 2, 2), torch.zeros(2, 2, 2)


def test_only_a_tiled_dataset_gets_the_window():
    """An untiled one has nothing to reuse, so it keeps torch's own shuffling and this
    change is invisible to it."""
    tiled = make_loader(_Tiled(10, 4), batch_size=2, shuffle=True)
    plain = make_loader(_Tiled(10, 1), batch_size=2, shuffle=True)

    assert isinstance(tiled.sampler, TileShuffle)
    assert not isinstance(plain.sampler, TileShuffle)


def test_nothing_is_shuffled_when_nothing_asked_for_it():
    """The one that matters most. `predict` and `evaluate` build their loaders with
    `shuffle=False` because **stitching depends on tiles arriving in the order they were
    cut** - a sampler leaking in here would reassemble every tiled prediction wrongly, and
    the picture would look plausible."""
    loader = make_loader(_Tiled(6, 4), batch_size=4, shuffle=False)

    assert not isinstance(loader.sampler, TileShuffle)
    assert list(torch.utils.data.SequentialSampler(range(24))) == list(loader.sampler)


@pytest.mark.parametrize("window", [1, 3, 8, 100])
def test_any_window_still_covers_the_set(window):
    """Including one wider than the dataset, which is what a large cache and a small split
    produce together."""
    sampler = TileShuffle(samples=7, tiles=3, window=window)
    assert sorted(sampler) == list(range(21))
