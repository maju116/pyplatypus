"""Samples on disk, presented as (image, instance target) pairs.

The sibling of `SegmentationDataset` and deliberately not a subclass of it. The two share
the shape of the job - read, resize, tile, augment, hand back channels-last arrays - and
almost none of the work: there is no colormap, no region vocabulary, no volume reader and
no spacing here, because a directory of per-object files names no classes and instances are
2-D (`InstanceModel.instances_are_two_dimensional` refuses the rest). What is shared is
reached as functions, `read_image` and `tile`, which both datasets call.

**The ordering is the whole content of this file**: the label map is resized and tiled and
augmented as integers, and the target is built from it afterwards. A target that encodes
direction cannot be transformed alongside the image - `pyplatypus/instances/targets.py` has
the measurement - so the one thing to preserve here is that `instance_target` is the last
call in `__getitem__`.

`unite_masks()` is what the semantic dataset does with the same files, and it is where
instance identity is thrown away: the element-wise maximum of 375 nuclei is one mask with
375 nuclei in it. Nothing new has to be collected to train on instances; the files only
have to be read differently, which is `label_map`'s job and this is where it is called.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from pyplatypus.data.augmentation import Augmenter
from pyplatypus.data.images import read_image, tile, to_float
from pyplatypus.data.paths import Sample
from pyplatypus.errors import PlatypusError
from pyplatypus.instances.metrics import label_map
from pyplatypus.instances.targets import instance_target, resize_labels
from pyplatypus.spec.instances import InstanceData, InstanceModel


class InstanceDataError(PlatypusError):
    kind = "instance_data_error"


@dataclass(frozen=True)
class ObjectReport:
    """What a split's objects look like, read from a sample of its cases.

    `objects` is counted **after** the resize, which is the number that matters and not the
    one on disk: at a small `input_shape` a nucleus of four pixels can vanish, and a split
    whose objects have all vanished trains against an empty target while every loss and
    every metric stays finite. `empty` is how many cases came back with no object at all.
    """

    cases_checked: int
    total_cases: int
    objects: int
    empty: int
    smallest: int
    largest: int


class InstanceDataset:
    """Images and their per-object masks, as (image, target) pairs.

    When the model tiles, one source image becomes several examples, so `len()` is samples x
    tiles and indexing walks the tiles of a sample before moving on - the same contract
    `SegmentationDataset` has, because `make_loader` and `TileShuffle` read it.
    """

    def __init__(
        self,
        samples: tuple[Sample, ...],
        model: InstanceModel,
        data: InstanceData,
        *,
        augmenter: Augmenter | None = None,
        only_images: bool = False,
        cache_size: int = 8,
    ):
        self.samples = samples
        self.model = model
        self.data = data
        self.augmenter = augmenter
        self.only_images = only_images
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()
        #: How many decoded samples are kept; the loader sizes its shuffle window from it.
        self.cache_size = max(1, cache_size)

    @property
    def tiles_per_sample(self) -> int:
        return self.model.tiles_per_image

    def __len__(self) -> int:
        return len(self.samples) * self.tiles_per_sample

    # ------------------------------------------------------------------ reading
    def _read_image(self, sample: Sample) -> np.ndarray:
        if len(sample.images) > 1:
            raise InstanceDataError(
                f"case '{sample.key}' has {len(sample.images)} image files. One file per "
                "case is what this task reads; `channels_from`, which is how the semantic "
                "task names one file per channel, has no instance equivalent yet - the "
                "classes here come from the objects rather than from the files."
            )
        return read_image(
            sample.image,
            channels=self.model.channels,
            size=self.model.load_shape,
            dicom_window=self.data.window,
        )

    def _read_labels(self, sample: Sample) -> np.ndarray:
        """Every object of one case as one label map, at the model's loading size.

        **The label map is built at the files' own resolution and resized afterwards**, not
        the other way round. Resizing each object's mask first and labelling the stack after
        would let two objects claim one pixel - nearest-neighbour on two separate masks has
        no way to agree - and `label_map` refuses overlap, so a dataset that measured zero
        overlapping objects on disk would start raising on its own data. One integer map
        going through one resize cannot produce an overlap at all.
        """
        # Discovery refuses a case with an empty label directory before a dataset exists,
        # so this fires only for samples a caller built by hand - which an example script
        # assembling the pieces does. Kept rather than left to `masks[0]`, which would say
        # `list index out of range`.
        if not sample.masks:
            raise InstanceDataError(
                f"case '{sample.key}' has no mask files. This task reads one file per "
                f"object from '{self.data.subdirs[1]}/', which is the Data Science Bowl's "
                "own layout; a case with none of them cannot be trained or scored against."
            )
        masks = [read_image(path, channels=1, nearest=True)[..., 0] > 0 for path in sample.masks]
        return resize_labels(label_map(masks), self.model.load_shape)

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray | None]:
        """One source sample at `load_shape`, cached because every tile asks again."""
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        sample = self.samples[index]
        image = self._read_image(sample)
        labels = None if self.only_images else self._read_labels(sample)

        self._cache[index] = (image, labels)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return image, labels

    # ------------------------------------------------------------------ indexing
    def labels_of(self, index: int) -> np.ndarray:
        """One **source** sample's label map at the loading grid, untiled and unaugmented.

        What the instance score is computed against, which is why it is not the array
        `__getitem__` builds its target from: scoring happens on the whole image after the
        tiles are stitched, because an object crossing a seam would otherwise be counted
        twice, and it happens on un-augmented data, because measuring a model on distorted
        data measures the distortion.
        """
        _, labels = self._load(index)
        if labels is None:
            raise InstanceDataError(
                "this dataset was built with only_images=True, so it read no objects to "
                "score against"
            )
        return labels

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray | None]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is outside 0..{len(self) - 1}")

        sample_index, tile_index = divmod(index, self.tiles_per_sample)
        image, labels = self._load(sample_index)

        if self.model.splits is not None:
            image = tile(image, self.model.splits)[tile_index]
            if labels is not None:
                labels = tile(labels[..., None], self.model.splits)[tile_index][..., 0]

        if self.augmenter is not None:
            image, labels = self.augmenter(image, labels)

        image = to_float(image)
        if labels is None:
            return image, None
        # Last, and that is the contract: every geometric step has already happened, so the
        # vectors and the rim are built from where the objects actually are.
        return image, instance_target(
            labels,
            self.model.signal,
            n_class=self.data.n_class,
            **self.model.target_settings,
        )

    def foreground_of(self, target: np.ndarray) -> np.ndarray:
        """Which voxels the target is defined on, for a loss that scores the foreground.

        Asked of the dataset rather than computed from the array, because the answer
        depends on which channels are classes: `onehot_to_foreground` reads channel 0 as
        background, which for a `flows` target is the row component of a vector field. Here
        the class block starts at `semantic_from` and its first channel is background.

        Never read off the vector channels, for the reason `instance_target` states: a flow
        field is exactly zero at every sink, so its non-zero voxels are the foreground minus
        precisely the seeds.
        """
        background = np.asarray(target)[..., self.model.separation.semantic_from]
        return (background < 0.5).astype(np.float32)[..., None]

    # ------------------------------------------------------------------ inspection
    def inspect_objects(self, limit: int = 20) -> ObjectReport:
        """Count the objects a sample of cases actually yields, after the resize.

        A sample rather than every case, for the reason the mask check reads a sample: this
        runs before training and 670 cases of 375 files each is a minute of reading to
        answer a question twenty cases answer.
        """
        checked = self.samples[: max(1, limit)]
        counts = [int(self.labels_of(index).max()) for index in range(len(checked))]
        return ObjectReport(
            cases_checked=len(counts),
            total_cases=len(self.samples),
            objects=sum(counts),
            empty=sum(1 for count in counts if count == 0),
            smallest=min(counts) if counts else 0,
            largest=max(counts) if counts else 0,
        )
