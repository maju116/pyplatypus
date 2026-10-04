"""The one place numpy becomes torch.

The data layer is channels-last because that is what PIL and albumentations speak; torch
wants channels-first. The transpose happens here, once, at the boundary - and it is
rank-generic, so a volume crosses the same way an image does.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pyplatypus.data.dataset import SegmentationDataset
from pyplatypus.data.detection import DetectionDataset


def to_channels_first(array: np.ndarray) -> torch.Tensor:
    """(*spatial, C) -> (C, *spatial), for any number of spatial dimensions."""
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(array, -1, 0)))


class TorchSegmentationDataset(Dataset):
    """A thin adapter. All the work already happened in `SegmentationDataset`."""

    def __init__(self, base: SegmentationDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, mask = self.base[index]
        if mask is None:
            return to_channels_first(image)
        return to_channels_first(image), to_channels_first(mask)


def make_loader(base: SegmentationDataset, *, batch_size: int = 8, shuffle: bool = False,
                num_workers: int = 0, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        TorchSegmentationDataset(base),
        batch_size=batch_size,
        shuffle=shuffle,
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
        return (to_channels_first(image),
                tuple(torch.from_numpy(t) for t in targets))


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


def make_detection_loader(base: DetectionDataset, *, batch_size: int = 8,
                          shuffle: bool = False, num_workers: int = 0,
                          drop_last: bool = False) -> DataLoader:
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
