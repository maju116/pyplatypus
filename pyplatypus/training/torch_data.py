"""The one place numpy becomes torch.

The data layer is channels-last because that is what PIL and albumentations speak; torch
wants channels-first. The transpose happens here, once, at the boundary - and it is
rank-generic, so a volume crosses the same way an image does.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from pyplatypus.data.dataset import SegmentationDataset
from pyplatypus.data.detection import DetectionDataset
from pyplatypus.data.masks import signed_distance


def to_channels_first(array: np.ndarray) -> torch.Tensor:
    """(*spatial, C) -> (C, *spatial), for any number of spatial dimensions."""
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(array, -1, 0)))


class TorchSegmentationDataset(Dataset):
    """A thin adapter. All the work already happened in `SegmentationDataset`.

    With `with_distance` it also computes the signed distance map of the mask, which is
    **why this is here and not in the loss**. Measured on this machine, one Euclidean
    distance transform of a compact lesion costs 5.3 ms at 256x256, 23.4 ms at 64x64x32
    and 709.8 ms at 128^3 - against an epoch of half a second for a small 3D U-Net. In the
    loss that would dominate training and make the largest volumes impossible; here it
    runs in the loader's workers, in parallel, behind the GPU.

    It is recomputed every epoch rather than cached, and deliberately: augmentation moves
    the mask, so a map cached against a sample index would describe a shape that is no
    longer there - and it would do it silently, since the map is never looked at.
    """

    def __init__(self, base: SegmentationDataset, *, with_distance: bool = False):
        self.base = base
        self.with_distance = with_distance

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, mask = self.base[index]
        if mask is None:
            return to_channels_first(image)
        if not self.with_distance:
            return to_channels_first(image), to_channels_first(mask)
        distance = signed_distance(mask, spacing=self._spacing())
        return (to_channels_first(image), to_channels_first(mask), to_channels_first(distance))

    def _spacing(self):
        """Millimetres per voxel if there is such a thing, and None if there is not.

        Only `target_spacing` gives one: it resamples every sample onto a common grid, so
        one number describes them all. Without it each sample is resized to `input_shape`
        independently, which means the physical size of a voxel differs per sample *and*
        has been changed by the resize - carrying a native spacing through would attach a
        number from before the distortion to a mask from after it.

        So the map is in millimetres when the data is on a common grid and in voxels
        otherwise, which is the only honest pair of answers.
        """
        spacing = getattr(self.base.data, "target_spacing", None)
        if spacing is None:
            return None
        # The mask's spatial axes, in the mask's own order; `target_spacing` is 3D.
        return tuple(float(v) for v in spacing)


class TileShuffle(Sampler[int]):
    """Shuffle a tiled dataset in windows, so a decoded sample serves all of its tiles.

    Shuffling over tile indices defeats the dataset's cache. Measured on FIVES - 2048-pixel
    retinas cut into sixteen - the loader costs **5.2 times more** shuffled than in order,
    49 ms a tile against 254, because a sample decodes in 210 ms and a cache of eight cannot
    hold a working set of six hundred.

    Emitting one sample's tiles together is the other extreme and makes a batch sixteen
    views of one retina, which is not the same gradient.

    So: take a window of `window` samples, shuffle all of their tiles together, move on.
    Every read then serves every tile of its sample, and a batch still draws from `window`
    different sources. The window is the cache size, because a wider one would evict a
    sample while its own tiles were still being asked for.

    Reproducible for the same reason the rest of the run is: it draws from torch's global
    generator, which `seed` sets.
    """

    def __init__(self, samples: int, tiles: int, window: int):
        self.samples = samples
        self.tiles = tiles
        self.window = max(1, window)

    def __len__(self) -> int:
        return self.samples * self.tiles

    def __iter__(self):
        order = torch.randperm(self.samples).tolist()
        for start in range(0, len(order), self.window):
            block = order[start : start + self.window]
            indices = [s * self.tiles + t for s in block for t in range(self.tiles)]
            for position in torch.randperm(len(indices)).tolist():
                yield indices[position]


def make_loader(
    base: SegmentationDataset,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    with_distance: bool = False,
) -> DataLoader:
    # A tiled dataset is shuffled in windows rather than wholesale; see `TileShuffle`. An
    # untiled one has nothing to reuse, so it takes torch's own shuffling and this changes
    # nothing for it.
    sampler = None
    if shuffle and base.tiles_per_sample > 1:
        sampler = TileShuffle(
            samples=len(base) // base.tiles_per_sample,
            tiles=base.tiles_per_sample,
            window=base.cache_size,
        )

    return DataLoader(
        TorchSegmentationDataset(base, with_distance=with_distance),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


class TorchDetectionDataset(Dataset):
    """A thin adapter, like `TorchSegmentationDataset`. One image, three targets."""

    def __init__(self, base: DetectionDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, targets = self.base[index]
        if targets is None:
            return to_channels_first(image)
        # The targets stay as they are: a YOLOv3 target is (rows, cols, anchors, 5 +
        # classes), where the last axis is not channels but a record per anchor. Moving
        # it would be meaningless, and the loss indexes it where it is.
        return (to_channels_first(image), tuple(torch.from_numpy(t) for t in targets))


def _collate_detection(batch):
    """Stack the images and each grid separately.

    torch's default collate would do this, but it recurses through the tuple and the
    message when something is the wrong shape names no grid. Three lines here beats
    reading a traceback out of `default_collate`.
    """
    if torch.is_tensor(batch[0]):
        return torch.stack(batch)
    images = torch.stack([row[0] for row in batch])
    grids = len(batch[0][1])
    targets = [torch.stack([row[1][grid] for row in batch]) for grid in range(grids)]
    return images, targets


def make_detection_loader(
    base: DetectionDataset,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        TorchDetectionDataset(base),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=_collate_detection,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
