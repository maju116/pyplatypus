"""One sample in, one training example out.

Numpy only. The torch `Dataset` wrapper is a dozen lines and arrives in step 3; keeping
it out of here means this layer stays fast to test and usable on its own.

Arrays are channels-last throughout, because that is what albumentations and PIL speak.
The torch adapter transposes once, at the boundary.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from pyplatypus.data.augmentation import Augmenter
from pyplatypus.data.images import read_image, tile, to_float
from pyplatypus.data.masks import (
    classes_to_onehot,
    colours_to_classes,
    labels_to_classes,
    unite_masks,
)
from pyplatypus.data.paths import Sample
from pyplatypus.errors import PlatypusError
from pyplatypus.spec.data import SegmentationData
from pyplatypus.spec.models import SegmentationModel


class DataError(PlatypusError):
    kind = "data_error"


def _is_series(paths: tuple) -> bool:
    """Several files that are all DICOM: one volume, arriving a slice at a time."""
    from pyplatypus.data.dicom import looks_like_dicom

    return len(paths) > 1 and all(looks_like_dicom(p) for p in paths)


class SegmentationDataset:
    """Samples on disk, presented as (image, one-hot mask) pairs.

    When the model tiles, one source image becomes several examples, so `len()` is
    samples x tiles and indexing walks the tiles of a sample before moving on.
    """

    def __init__(self, samples: tuple[Sample, ...], model: SegmentationModel,
                 data: SegmentationData, *, augmenter: Augmenter | None = None,
                 only_images: bool = False, cache_size: int = 8):
        self.samples = samples
        self.model = model
        self.data = data
        self.augmenter = augmenter
        self.only_images = only_images
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()
        self._cache_size = max(1, cache_size)

    @property
    def tiles_per_sample(self) -> int:
        return self.model.tiles_per_image

    def __len__(self) -> int:
        return len(self.samples) * self.tiles_per_sample

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Read one source sample at `load_shape`, cached because every tile asks again."""
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        sample = self.samples[index]
        size = self.model.load_shape
        image = self._read(sample.images, channels=self.model.channels, size=size)

        classes: np.ndarray | None = None
        if not self.only_images:
            # Nearest, always: interpolating a mask invents values that belong to no class
            # and would quietly become background.
            mask_channels = 1 if self.data.label_map else 3
            if self.model.rank == 3 and _is_series(sample.masks):
                masks = [self._read(sample.masks, channels=mask_channels, size=size,
                                    nearest=True)]
            else:
                masks = [read_image(p, channels=mask_channels, size=size, nearest=True)
                         for p in sample.masks]
            united = unite_masks(masks)
            classes, _ = self._to_classes(united)

        self._cache[index] = (image, classes)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return image, classes

    def _read(self, paths: tuple, *, channels: int, size: tuple[int, ...],
              nearest: bool = False) -> np.ndarray:
        """Read one sample's files, which may be a stack of DICOM slices.

        A 3D model and several files per sample means those files are one volume. The rank
        decides rather than a setting, for the same reason rank itself is derived: a 3D model
        could not consume several separate volumes anyway.
        """
        if self.model.rank == 3 and _is_series(paths):
            from pyplatypus.data.dicom_series import read_dicom_series

            return read_dicom_series(list(paths), window=self.data.window,
                                     channels=channels, size=size, nearest=nearest)
        if self.model.rank == 3 and len(paths) > 1:
            raise DataError(
                f"sample has {len(paths)} files and the model is 3D, but they are not DICOM "
                "slices. Several volumes per sample - one per modality, say - is not "
                "supported yet; give one volume per sample."
            )
        return read_image(paths[0], channels=channels, size=size, nearest=nearest,
                          dicom_window=self.data.window)

    def _to_classes(self, mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Whichever way this dataset names its classes."""
        if self.data.label_map:
            return labels_to_classes(mask, self.data.labels)
        return colours_to_classes(mask, self.data.colormap)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray | None]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is outside 0..{len(self) - 1}")

        sample_index, tile_index = divmod(index, self.tiles_per_sample)
        image, classes = self._load(sample_index)

        if self.model.splits is not None:
            # Rank-generic: a 2D grid and a 3D patch grid are the same cut.
            image = tile(image, self.model.splits)[tile_index]
            if classes is not None:
                classes = tile(classes[..., None], self.model.splits)[tile_index][..., 0]

        if self.augmenter is not None:
            image, classes = self.augmenter(image, classes)

        image = to_float(image)
        if classes is None:
            return image, None
        return image, classes_to_onehot(classes, self.model.n_class)

    def colormap_coverage(self, limit: int = 20) -> float:
        """Fraction of mask voxels matching no colour, or no label value.

        A number near 1 means the colormap - or the list of labels - does not describe this
        dataset, which is the single most common way a segmentation run silently trains on
        nothing at all.
        """
        if self.only_images:
            raise DataError("there are no masks to check")
        unmatched = []
        mask_channels = 1 if self.data.label_map else 3
        for index in range(min(limit, len(self.samples))):
            sample = self.samples[index]
            masks = [self._read(sample.masks, channels=mask_channels,
                                size=self.model.load_shape, nearest=True)] \
                if self.model.rank == 3 and _is_series(sample.masks) else [
                    read_image(p, channels=mask_channels, size=self.model.load_shape,
                               nearest=True)
                    for p in sample.masks
                ]
            _, fraction = self._to_classes(unite_masks(masks))
            unmatched.append(fraction)
        return float(np.mean(unmatched))
