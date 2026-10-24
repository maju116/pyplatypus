"""One sample in, one training example out.

Numpy only. The torch `Dataset` wrapper is a dozen lines and arrives in step 3; keeping
it out of here means this layer stays fast to test and usable on its own.

Arrays are channels-last throughout, because that is what albumentations and PIL speak.
The torch adapter transposes once, at the boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from pyplatypus.data.augmentation import Augmenter
from pyplatypus.data.channels import match_channels
from pyplatypus.data.images import read_image, tile, to_float
from pyplatypus.data.masks import (
    classes_to_onehot,
    codes_to_onehot,
    colours_to_classes,
    labels_to_classes,
    labels_to_regions,
    masks_to_codes,
    onehot_to_foreground,
    region_vocabulary,
    unite_masks,
)
from pyplatypus.data.paths import Sample
from pyplatypus.errors import PlatypusError
from pyplatypus.spec.data import SegmentationData
from pyplatypus.spec.models import SegmentationModel


class DataError(PlatypusError):
    kind = "data_error"


#: How many masks to read before an early exit is allowed. One file's unmatched fraction is
#: not an estimate of anything.
_MINIMUM_SAMPLES = 5


def _spread(total: int, count: int) -> list[int]:
    """`count` indices over `0..total-1`, ordered so that any prefix still spans the range.

    Two things are wanted at once and they pull apart. The sample must cross the whole
    dataset, because real ones arrive sorted - by patient, by acquisition date, sometimes
    by class - and a prefix of a sorted listing is a biased sample: "this class is absent"
    drawn from one can mean only "the positives are later in the list". But the scan also
    wants to stop early when it has its answer, and stopping early on an in-order sweep
    reads only the beginning, which is the bias again.

    So the order bisects: first, last, middle, then the midpoints of what is left. Ten
    reads out of ten thousand samples still touch both ends and the middle. Deterministic,
    and needing no seed to be reproducible.
    """
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        positions = list(range(total))
    else:
        step = (total - 1) / (count - 1) if count > 1 else 0
        positions = sorted({round(i * step) for i in range(count)})

    order: list[int] = []

    def bisect(lo: int, hi: int) -> None:
        if lo > hi:
            return
        mid = (lo + hi) // 2
        order.append(positions[mid])
        bisect(lo, mid - 1)
        bisect(mid + 1, hi)

    if positions:
        order.append(positions[0])
        if len(positions) > 1:
            order.append(positions[-1])
        bisect(1, len(positions) - 2)
    seen = set()
    return [i for i in order if not (i in seen or seen.add(i))]


@dataclass(frozen=True)
class MaskReport:
    """What a colormap or a list of labels matched, over a sample of the masks."""

    unmatched: float
    present_classes: list[int]
    missing_classes: list[int]
    samples_checked: int
    total_samples: int


def _series_spacing(paths: tuple):
    from pyplatypus.data.dicom_series import series_spacing

    return series_spacing(list(paths))


def _is_series(paths: tuple) -> bool:
    """Several files that are all DICOM: one volume, arriving a slice at a time."""
    from pyplatypus.data.dicom import looks_like_dicom

    return len(paths) > 1 and all(looks_like_dicom(p) for p in paths)


class SegmentationDataset:
    """Samples on disk, presented as (image, one-hot mask) pairs.

    When the model tiles, one source image becomes several examples, so `len()` is
    samples x tiles and indexing walks the tiles of a sample before moving on.
    """

    def __init__(
        self,
        samples: tuple[Sample, ...],
        model: SegmentationModel,
        data: SegmentationData,
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
        # Nested regions read a label map into a vocabulary of its values and expand to one
        # binary channel per region at the very end. Built once: the pair has to agree, and
        # a table indexed with a vocabulary it was not built from still produces a plausible
        # target. None for every other naming - including `classes_from`, which is also
        # multilabel and has no label values to build a vocabulary from, so this asks the
        # field rather than the switch.
        self._region_values, self._region_table = (
            region_vocabulary([region.of for region in data.regions])
            if data.regions is not None
            else (None, None)
        )
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()
        #: How many decoded samples are kept. Public because the loader sizes its shuffle
        #: window from it: a window wider than the cache would evict a sample while its own
        #: tiles were still being asked for, which is the thing the cache exists to prevent.
        self.cache_size = max(1, cache_size)

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
        image = self._read(
            sample.images, channels=self.model.channels, size=size, as_channels=True, key=sample.key
        )

        classes: np.ndarray | None = None
        if not self.only_images:
            classes, _ = self._read_masks(sample, size)

        self._cache[index] = (image, classes)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return image, classes

    def _read(
        self,
        paths: tuple,
        *,
        channels: int,
        size: tuple[int, ...],
        nearest: bool = False,
        as_channels: bool = False,
        key: str | None = None,
    ) -> np.ndarray:
        """Read one sample's image files as one array.

        Three shapes of input, distinguished by what the data says rather than by guessing:
        one file; several DICOM files, which are the slices of one volume; and several files
        that are one channel each, which only happens when `channels_from` says so and says in
        which order.
        """
        if as_channels and self.data.channels_from is not None and len(paths) > 1:
            ordered = match_channels(paths, self.data.channels_from, key=key)
            if self.model.rank == 3:
                return self._read_volume_channels(ordered, size=size, nearest=nearest, key=key)
            # In 2D the files are bands of one scene and may legitimately arrive at different
            # resolutions - Sentinel ships 10 m and 20 m bands of the same tile - so each is
            # read at the wanted size and no geometry is being asserted.
            planes = [
                read_image(
                    path, channels=1, size=size, nearest=nearest, dicom_window=self.data.window
                )
                for path in ordered
            ]
            return np.concatenate(planes, axis=-1)

        if self.model.rank == 3 and _is_series(paths):
            from pyplatypus.data.dicom_series import read_dicom_series

            if self.data.target_spacing is not None and size is not None:
                return self._at_spacing(
                    read_dicom_series(
                        list(paths), window=self.data.window, channels=channels, nearest=nearest
                    ),
                    _series_spacing(paths),
                    size=size,
                    nearest=nearest,
                )
            return read_dicom_series(
                list(paths), window=self.data.window, channels=channels, size=size, nearest=nearest
            )

        if len(paths) > 1 and self.model.rank == 3:
            raise DataError(
                f"sample '{key}' has {len(paths)} files and the model is 3D, but they are "
                "neither DICOM slices nor named channels. For one file per channel - four MRI "
                "sequences per patient, say - set `channels_from` with one pattern per "
                "channel; the order has to be stated, because sorting the names gives an "
                "order that is reproducible and anatomically meaningless."
            )
        return self._read_one(paths[0], channels=channels, size=size, nearest=nearest)

    def _read_volume_channels(
        self, ordered: tuple, *, size: tuple[int, ...], nearest: bool, key: str | None
    ) -> np.ndarray:
        """Stack one volume per channel, checking they describe the same anatomy first.

        Read at native resolution and compared before anything is resized, because resizing
        each channel to the model's shape separately would *hide* a mismatch: two sequences
        that do not line up voxel for voxel would arrive the same size with their anatomy in
        different places, and nothing downstream could tell. In a volume the geometry is
        knowable, so it is checked rather than papered over.
        """
        from pyplatypus.data.volumes import volume_spacing

        planes, spacings = [], []
        for path in ordered:
            planes.append(
                read_image(path, channels=1, nearest=nearest, dicom_window=self.data.window)
            )
            if self.data.target_spacing is not None:
                spacings.append(volume_spacing(path))

        shapes = {plane.shape for plane in planes}
        if len(shapes) > 1:
            listed = ", ".join(
                f"{path.name}: {plane.shape[:3]}"
                for path, plane in zip(ordered, planes, strict=True)
            )
            raise DataError(
                f"the channels of sample '{key}' have different shapes ({listed}). Channels "
                "of one sample are measurements of the same anatomy and have to line up voxel "
                "for voxel; resizing them to match would leave the second one's anatomy in the "
                "wrong place. Register them first."
            )

        stacked = np.concatenate(planes, axis=-1)
        if size is None:
            return stacked
        if self.data.target_spacing is not None:
            if len({tuple(round(v, 4) for v in s) for s in spacings}) > 1:
                raise DataError(
                    f"the channels of sample '{key}' have different voxel spacing: "
                    f"{spacings}. Same shape and different spacing means they cover different "
                    "amounts of anatomy."
                )
            return self._at_spacing(stacked, spacings[0], size=size, nearest=nearest)

        from pyplatypus.data.volumes import resize_volume

        return resize_volume(stacked, size, nearest=nearest)

    def _read_one(
        self, path, *, channels: int, size: tuple[int, ...] | None, nearest: bool = False
    ) -> np.ndarray:
        """One file, resampled to the wanted voxel size when that was asked for.

        `size=None` means **as stored**: no resampling and no crop, which is the grid the
        annotation was drawn on and the only one every model in a comparison shares. It is
        what `evaluate(at="source")` scores against, and it has to skip `target_spacing`
        rather than merely skipping the crop - resampling is what moves a voxel.
        """
        if self.model.rank == 3 and self.data.target_spacing is not None and size is not None:
            from pyplatypus.data.volumes import volume_spacing

            volume = read_image(
                path, channels=channels, nearest=nearest, dicom_window=self.data.window
            )
            return self._at_spacing(volume, volume_spacing(path), size=size, nearest=nearest)
        return read_image(
            path, channels=channels, size=size, nearest=nearest, dicom_window=self.data.window
        )

    def _at_spacing(
        self, volume: np.ndarray, spacing, *, size: tuple[int, ...], nearest: bool
    ) -> np.ndarray:
        """Resample to `target_spacing`, then fit to `size` by cropping or padding.

        The order matters. Reading straight to `size` resizes each volume into the same box, so
        two scans covering different lengths of patient come out at different physical scale.
        Resampling fixes the millimetres per voxel; cropping or padding afterwards turns the
        varying shape that leaves into the one shape a network needs, without stretching away
        what the resampling just established.
        """
        from pyplatypus.data.volumes import crop_or_pad, resample_to_spacing

        resampled = resample_to_spacing(volume, spacing, self.data.target_spacing, nearest=nearest)
        return crop_or_pad(resampled, size)

    def _read_masks(self, sample, size: tuple[int, ...] | None) -> tuple[np.ndarray, float]:
        """One sample's masks as one integer per voxel, and what matched nothing.

        `size=None` reads them **as stored** - the grid the annotation was drawn on, with no
        resampling and no crop. That is what `evaluate(at="source")` compares against, and
        reading it through this method rather than through a second one of its own is
        deliberate: the three namings below are the part that must not exist twice.

        Nearest, always: interpolating a mask invents values that belong to no class and
        would quietly become background.

        **`classes_from` is the one naming whose files must not be united.** The other
        three take the element-wise maximum of a sample's mask files - the Data Science Bowl
        ships one per nucleus - and that maximum is exactly where per-class identity is
        destroyed, which is the whole reason this field needed a reader rather than a
        validator. So the files are matched to their patterns, read in that order and coded
        a bit per class instead.

        One method rather than the two copies `_load` and `inspect_masks` had, because the
        third naming would have made it three: the read is not where either of them has
        anything of its own to say.
        """
        if self.data.classes_from is not None:
            ordered = match_channels(
                sample.masks, self.data.classes_from, key=sample.key, what="class"
            )
            # `_read_one` rather than `read_image`, so a volume per class is resampled to
            # `target_spacing` the way the image beside it is - the bug the rank-3 branch
            # below was written for, which this path would otherwise repeat.
            planes = [self._read_one(path, channels=1, size=size, nearest=True) for path in ordered]
            return masks_to_codes(planes)

        mask_channels = 1 if self.data.label_map else 3
        if self.model.rank == 3:
            # The same reader as the image, always. Sending masks down a different path was
            # a real bug: with target_spacing the image was resampled and cropped while the
            # mask was merely resized, so every label sat beside the anatomy it was
            # labelling. Both go through one function now, and there is a test that the mask
            # covers the tissue.
            # Never `as_channels`: a mask is one labelling of one anatomy, and four
            # modalities do not come with four masks.
            masks = [
                self._read(
                    sample.masks,
                    channels=mask_channels,
                    size=size,
                    nearest=True,
                    key=sample.key,
                )
            ]
        else:
            # 2D keeps one file per object - the Data Science Bowl ships a mask per nucleus
            # - and unites them.
            masks = [
                read_image(p, channels=mask_channels, size=size, nearest=True) for p in sample.masks
            ]
        return self._to_classes(unite_masks(masks))

    def _to_classes(self, mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Whichever way this dataset names its classes.

        All three return one integer per voxel, which is the point: tiling and augmentation
        operate on a single-channel array and neither has to know which naming produced it.
        For `regions` the integer indexes the region vocabulary rather than a class, and 0
        means a value no region mentions.
        """
        if self.data.multilabel:
            return labels_to_regions(mask, self._region_values)
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
        return image, self.to_target(classes)

    def to_target(self, classes: np.ndarray) -> np.ndarray:
        """One integer per voxel expanded into one channel per class.

        Three ways, decided by how this dataset names its classes - and a method rather
        than a tail of `__getitem__` because scoring on the source grid needs the same
        expansion, and which of the three applies is a property of the data rather than of
        where the scoring happens.
        """
        if self.data.classes_from is not None:
            # Bits rather than a table: a table of combinations would need two to the power
            # of the classes, where the regions' table needs one row per label value.
            return codes_to_onehot(classes, self.data.n_class)
        if self.data.regions is not None:
            # The exclusive case is `np.eye(n_class)[classes]`; this is the same indexing
            # with a table whose rows are region membership instead of the identity, so a
            # voxel can land in several channels at once - or, from row 0, in none.
            return self._region_table[classes]
        return classes_to_onehot(classes, self.data.n_class)

    def target_as_stored(self, sample) -> np.ndarray:
        """A sample's target on the grid its files hold: channels last, one per class.

        No resampling, no crop and no tiling - the grid the annotation was drawn on, which
        is the one grid every model in a comparison shares. `evaluate(at="source")` scores
        against this.
        """
        classes, _ = self._read_masks(sample, None)
        return self.to_target(classes)

    def foreground_of(self, mask: np.ndarray) -> np.ndarray:
        """Which voxels the mask assigns to something, for a loss that needs to know.

        A method rather than the loader calling `onehot_to_foreground` directly, because
        the answer depends on which channels are classes and the instance dataset's are not
        in the same place: a `flows` target's channel 0 is the row component of a vector
        field, and read as background it would call the whole image foreground. One question
        asked of whichever dataset is in the loader - the `EngineBase` lesson applied one
        layer down, where two datasets had begun to need two answers.
        """
        return onehot_to_foreground(mask, multilabel=self.data.multilabel)

    def colormap_coverage(self, limit: int = 20) -> float:
        """Fraction of mask voxels matching no colour, or no label value.

        Kept as it was because the R surface calls it. `inspect_masks` is the fuller
        answer, and the one worth acting on: see what this number cannot see, below.
        """
        return self.inspect_masks(limit=limit).unmatched

    def inspect_masks(self, limit: int = 20) -> MaskReport:
        """What the colormap, or the list of labels, actually matches in these masks.

        Two findings, and the second is the one that matters.

        `unmatched` is the fraction of mask voxels matching no entry, which fall back to
        class 0. Near 1 it says the colormap does not describe this dataset at all.

        **But it scales with the size of the thing being segmented, so on its own it
        cannot see the failure that matters most.** Measured on synthetic masks whose
        foreground is pure white and a colormap asking for VOC red instead:

            foreground 20.0% of the image -> unmatched 19.83%
            foreground  5.0%              -> unmatched  4.96%
            foreground  0.6%              -> unmatched  0.61%

        The last is a colormap that matches none of the labelled tissue, and it is
        indistinguishable from the 0.72% that JPEG compression leaves around the edges of
        a correct mask. A small lesion is the ordinary medical case, so the quietest
        signal belongs to the most dangerous mistake.

        `missing_classes` does not scale. A declared class that appears in no mask at all
        is provable: its channel in the target is zero everywhere, so the loss has nothing
        to push towards and the model cannot be shown what it is meant to find. In the
        same measurement the correct colormap gave per-class counts of [15984, 400] and
        the wrong one gave [16384, 0] - 400 pixels against exactly none, at any lesion
        size.

        **With `regions`, read `unmatched` differently and `missing_classes` the same way.**
        Nested regions have no background entry on purpose, so every background voxel counts
        as unmatched and the fraction sits near 1 on any dataset where the thing being found
        is small - on BraTS it is the correct answer and it looks like the catastrophic one.
        `missing_classes` keeps its meaning exactly, and gains the failure `regions` invites:
        `of: [3]` against data holding `4` is a region matching nothing, which trains to
        predict nothing and scores a Dice of 0 while every other number looks ordinary. It is
        reported here by region index.
        """
        if self.only_images:
            raise DataError("there are no masks to check")
        unmatched = []
        seen: set[int] = set()
        total = len(self.samples)
        wanted = set(range(self.data.n_class))
        read = 0
        for index in _spread(total, min(limit, total)):
            # Stop once every declared class has turned up and enough files have been read
            # for the unmatched figure to mean something. The good case then costs a
            # handful of reads, and the effort goes where there is doubt - on Data Science
            # Bowl, where each sample carries one mask file per nucleus, scanning fifty
            # samples took 3.8 seconds and the first few answer the question.
            if read >= _MINIMUM_SAMPLES and not (wanted - seen):
                break
            sample = self.samples[index]
            classes, fraction = self._read_masks(sample, self.model.load_shape)
            unmatched.append(fraction)
            if self.data.classes_from is not None:
                # The codes are bit sets, so one OR over the distinct values says which
                # classes appear anywhere - and a pattern matching a file that is empty in
                # every sample read is exactly what `missing_classes` is for: a channel of
                # zeros trains to predict nothing and scores a Dice of 0 while every other
                # number looks ordinary.
                together = int(np.bitwise_or.reduce(np.unique(classes)))
                seen.update(index for index in range(self.data.n_class) if together >> index & 1)
            elif self.data.regions is not None:
                # `classes` indexes the vocabulary, not the regions, so the table says which
                # regions those values put a voxel in. Without this step `missing_classes`
                # would compare vocabulary positions against region positions and report a
                # plausible, wrong answer.
                present = self._region_table[np.unique(classes)].any(axis=0)
                seen.update(int(region) for region in np.flatnonzero(present))
            else:
                seen.update(int(value) for value in np.unique(classes))
            read += 1
        return MaskReport(
            unmatched=float(np.mean(unmatched)) if unmatched else 0.0,
            present_classes=sorted(seen),
            missing_classes=sorted(wanted - seen),
            samples_checked=read,
            total_samples=total,
        )
