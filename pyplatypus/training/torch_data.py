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
