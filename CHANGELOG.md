# Change Log
All notable changes to this project will be documented in this file.

## [0.3.0a10]

*2026-10-02*

### Added

 - **Pretrained encoders.** `encoder: resnet34` builds a timm backbone as the contracting path and
   `pretrained: true` is a separate flag that loads its ImageNet weights - separate on purpose, so
   naming a backbone never reaches the network on a machine that has none. `timm` is the optional
   `encoders` extra
 - `freeze_encoder: 5` holds the transferred layers still for that many epochs, and
   `encoder_learning_rate` gives them a rate of their own. Both act on the layers that actually
   arrived pretrained and not on the full-resolution stage added in front of them, which starts
   random and is the one part that must keep learning. Parameter groups are split by object
   identity, so no rename can misfile one
 - Freezing sets `eval()` as well as `requires_grad`, and does it *after* `model.train()`: a frozen
   BatchNorm otherwise rewrites its running statistics from every batch it sees, which changes the
   very thing the freeze was protecting
 - `examples/compare_pretrained_encoders.py` and the measurement it produced, in the README: three
   datasets, one per modality, three seeds each. **The mechanisms work and the transfer does not.**
   Use `freeze_encoder` rather than `encoder_learning_rate` alone - on Kvasir a tenth of the rate
   is the worst configuration in the table. Expect convergence in half the epochs rather than a
   better score. A bigger encoder earns its place where the target is thin, pretrained or not: on
   SIIM-ACR every ResNet variant roughly doubles the built-in encoder, whose seed spread there is
   larger than its own mean

### Changed

 - `verify_encoder` checks a supplied encoder **by measurement** rather than by its declaration: one
   tiny probe at build time confirms the level count, that `channels` matches what comes back, that
   level 0 is at input resolution, and that every level halves. The seam had an unstated contract -
   an encoder whose stem strides by 2, which is every ImageNet backbone, was accepted in silence and
   produced masks at half the size of the ones scoring them, surfacing much later as a bare tensor
   size error from inside the loss
 - Levels are chosen from a backbone by downsampling factor rather than position, so `vgg16`'s
   full-resolution level is used directly and the patch-based backbones (`convnext_*`, `swin_*`),
   whose features start at 1/4, are refused by name with the reason instead of having a 1/2 level
   invented for them
 - CI installs the `encoders` extra. An optional extra CI does not install is a feature shipped
   untested behind a green run, because its tests skip rather than fail

### Refused

 - `pretrained` without `encoder`, which would otherwise read as transfer learning and do nothing
 - `encoder` at rank 3, caught while the specification is read so a 3D run fails before anything is
   downloaded. ImageNet is images; there is nothing to transfer to a volume
 - `blocks` deeper than the backbone has stages, and backbone names timm does not know

## [0.3.0a9]

*2026-09-27*

### Added

 - `examples/compare_dsbowl_architectures.py`: all four architectures in one specification, on one
   split, reported with the size and the cost beside the score. Dice alone cannot decide which
   weights are worth publishing - a model scoring 0.001 higher for twice the epoch time is a
   different trade, not a better choice
 - The comparison itself, in the README and the model card, because the result is the useful part:
   **the spread across the four architectures is 0.0030 while the spread across images within one
   of them is 0.054.** Re-running the same U-Net with another seed moves the mean by 0.0012, 40% of
   the whole between-architecture spread. On this problem the architecture is not where the result
   comes from, and that is worth an hour of somebody's compute to learn rather than a fortnight of
   swapping decoders
 - It is also why `dsbowl-unet` remains the only published entry: four names meaning the same thing
   would be four permanent promises nobody needed

## [0.3.0a8]

*2026-09-27*

### Added

 - The first published weights: **`dsbowl-unet`**, a U-Net at 256x256 that separates cell nuclei
   from background in light microscopy. Trained on BBBC038v1 - the 2018 Data Science Bowl images,
   CC0, taken from the Broad Bioimage Benchmark Collection rather than the Kaggle mirror. Dice
   0.9205 mean over 134 held-out images, sd 0.053, median 0.9314, **worst case 0.7240**; the five
   worst images are named in the sidecar
 - It is **semantic, not instance**: touching nuclei come back as one region, so the numbers are
   not comparable to the challenge's own leaderboard, which scored instances. The registry
   description and the model card both say so, because someone counting nuclei with these weights
   would otherwise get a wrong answer with no warning
 - Pinned to commit `5e035c6a` of `maju116/platypus-weights`. Verified end to end against the real
   Hub: the name resolves, the file lands in the Hugging Face cache, the sidecar travels with it,
   and the reloaded weights score 0.9201 on the same validation split. A second call takes two
   milliseconds and no network

## [0.3.0a7]

*2026-09-27*

### Added

 - Published weights by name. Three forms, in the order they will be used: `dsbowl-unet` from the
   registry, `hf://owner/repo/file.safetensors@commit` for anything else on the Hub, and a local
   path. A registry name carries the **commit** it was published at, and the `hf://` form requires
   one - weights that change under a stable name are the worst kind of irreproducibility, since
   the code is identical and the result is not
 - A **sidecar** beside every published file, recording the architecture, input shape, channels,
   classes, blocks and filters the weights were trained with, plus provenance. Loading refuses a
   model they do not belong to *before* touching torch: a shape mismatch torch would catch anyway,
   but weights trained on a different colormap with the same class count load cleanly and predict
   nonsense
 - `Engine.export_weights()` writes safetensors plus that sidecar, and takes arbitrary extra
   fields - the dataset, its licence, the citation, the measured validation distribution. Weights
   whose provenance is only in somebody's memory cannot be used by anybody else
 - `safetensors` is a hard dependency; `huggingface_hub` is the optional `hub` extra, since most
   runs never fetch weights and an air-gapped one cannot. Asking for a name without it gives a
   message saying which command fixes it
 - `examples/publish_dsbowl_weights.py` trains the first published weights from **BBBC038v1 at
   the Broad Institute**, not the Kaggle mirror. The images are identical; the Kaggle copy is
   governed by competition rules accepted at download, while BBBC038v1 carries an explicit CC0
   waiver, and weights are a derivative of the data behind them

### Notes

 - Downloads land in the Hugging Face cache (`~/.cache/huggingface/hub`, or `HF_HOME`), which sits
   outside any virtual environment - so weights survive reticulate rebuilding its uv environment,
   and a pinned commit needs no network call once cached. Air-gapped sites now have two caches to
   warm: that one and `~/.cache/uv`

## [0.3.0a6]

*2026-09-27*

### Added

 - `predict(space="source")`: each prediction mapped back onto the grid of the file it was
   computed from, undoing the resampling and the crop. Until now a prediction came back on the
   model's grid only, which meant a mask that could be scored but not laid over the patient's
   scan - `save_volumes()` refused to write it, correctly, since a mask with the wrong geometry
   lands in the wrong place
 - It returns a **list**, not an array, and that is the honest type: sources differ in size, a
   stack needs one shape, and resizing them to match would produce a mask describing anatomy it
   was not computed from. `space="model"` keeps the stacked array and remains the default
 - The inverse is built from the forward path rather than guessed: crop or pad to the shape
   resampling produced, then resize to the source's own shape, which lands on the source shape
   by construction rather than by rounding. Works with or without `target_spacing`, and at both
   ranks
 - `spatial_shape()`, `volume_shape()` and `series_shape()` read a source's size from headers
   alone - for volumes the *canonical* shape, matching the array the reader returns rather than
   the file's own axis order

### Known losses, documented rather than hidden

 - Interpolating probabilities smooths them, so a boundary returns slightly softer than the
   model drew it
 - Where the forward crop cut anatomy away, the inverse pads it with background. That padding
   means **not examined**, not **nothing there**. Give the model an `input_shape` that covers the
   anatomy when the distinction matters

## [0.3.0a5]

*2026-09-26*

### Added

 - Augmentation in 3D. Volumes go to albumentations through `volume` and `mask3d` rather than
   `image` and `mask`, and one set of parameters applies to the whole volume - a per-slice
   augmentation would tear the anatomy apart, and there is a test that every slice gets the
   same geometry
 - Support is uneven and that is handled rather than hidden. Of the transforms in albumentations
   2.0.8, 97 can take a volume - `Affine`, `ElasticTransform`, `D4`, `CubicSymmetry`, the flips,
   the blurs, the 3D-native crops - while others raise from inside the library, `GaussNoise` and
   `ChannelDropout` as `KeyError: 'images'`, which is not a sentence anyone can act on. Each
   transform in a 3D spec is tried against a small probe volume while the pipeline is built, and
   an unsupported one is named then rather than forty minutes into training
 - Probed per step rather than as a pipeline, so the message names the offending transform
   instead of leaving a user to bisect eight of them
 - `available_transforms(rank=3)` lists what can be used on volumes. Transforms that need
   arguments cannot be probed with defaults and are kept rather than dropped - `CenterCrop3D` is
   one of them, and it is written for volumes. "Could not check" is not "does not work"

### Fixed

 - The probe forces `p=1`. Without it the check asked its question only half the time, because
   most transforms default to `p=0.5`: an unsupported transform passed roughly every other run
   and crashed mid-epoch instead. The first version of the tests was flaky for the same reason
   from the other side, and was caught by running them eight times rather than once

## [0.3.0a4]

*2026-09-26*

### Added

 - `channels_from`: one pattern per channel, in channel order, for datasets that keep one
   channel per file. BraTS ships four MRI sequences per patient - T1, T1 after contrast, T2,
   FLAIR - because different tumour structures are visible in different sequences; the 38-Cloud
   satellite set keeps its bands the same way. Works at both ranks
 - The order is **stated, never inferred.** Sorted, BraTS gives flair, t1, t1ce, t2: perfectly
   reproducible and anatomically meaningless. A model trained with FLAIR in channel one and
   then used on data whose channel one is T1 returns a plausible answer, with the right shape
   and the right range, and nothing downstream can detect it
 - Refusals, each naming the sample: a pattern matching no file, a pattern matching two (`_t1`
   matches `_t1.nii.gz` and `_t1ce.nii.gz` alike), two channels resolving to one file, and a
   file matched by no pattern - which left alone would be data the model never sees, and on a
   dataset where an extra sequence appears for some patients only, a difference between cases
 - Volume channels are compared **before** anything is resized, and a mismatch is an error.
   Resizing each channel to the model's shape separately would hide it: the channels would
   arrive the same size with their anatomy in different places. Spacing is checked too when
   resampling. In 2D the bands of one scene may legitimately differ in resolution - Sentinel
   ships 10 m and 20 m bands of one tile - so there no geometry is asserted
 - `channels_from` is cross-checked against every model's `channels`, so a mismatch is a
   specification error rather than a shape failure deep inside torch

## [0.3.0a3]

*2026-09-26*

### Added

 - `target_spacing` on the data section: resample every volume to a common voxel size, then
   centre-crop or pad to the model's `input_shape`. Without it a volume is resized into the
   box, and that is only harmless when two scans cover the same extent. Clinical scans do not:
   40 slices of 1 mm is 40 mm of patient and 40 slices of 2.5 mm is 100 mm, so resizing makes
   the same organ a different size in each and nothing in the data says so. Resampling fixes
   the millimetres per voxel - the quantity anatomy is measured in - and cropping or padding
   afterwards is what turns the varying shape into the one shape a network needs without
   stretching away what the resampling established
 - `resample_to_spacing()` and `crop_or_pad()`, centred, usable on their own
 - Measured on a 10 mm sphere: 4189 mm3 in theory, 4224 at 1 mm slices. The same sphere at
   2.5 mm slices reads 1688 voxels against the fine scan's 4224 - a 60% difference in what is
   physically one object - and 3760 after resampling, an 11% difference. The rest is honest:
   at 2.5 mm the caps were never measured, and resampling recovers no information an
   acquisition did not take

### Fixed

 - **Masks went down a different reading path from images.** With `target_spacing` the image
   was resampled and cropped while a single-file mask was merely resized, so every label sat
   beside the anatomy it was labelling - a misalignment that trains quietly and shows up as a
   model that cannot learn. Both now go through one function, and a test asserts the mask
   covers the tissue rather than merely having the right shape

## [0.3.0a2]

*2026-09-26*

### Added

 - Reading a DICOM series: a folder of slices becomes one volume, which is how data leaves a
   hospital. Four decisions, each of which produces a volume that trains without complaint
   when taken wrongly:
   - **order** comes from `ImagePositionPatient` projected onto the slice normal. Not the
     filename, which sorts slice 10 before slice 2, and not `InstanceNumber` either, which
     need only be unique within a series and is not required to follow the anatomy - it is
     the documented fallback when there is no geometry at all, and `Series.sorted_by` says
     which was used
   - **one series at a time.** A folder out of an archive usually holds several - a scout, a
     reconstruction, a phase - and `SeriesInstanceUID` separates them. Stacking two
     interleaves two anatomies at two resolutions
   - **one window for the whole series.** Slices of one series can record different windows;
     honoured slice by slice they produce a brightness gradient the scanner never measured
   - **canonical orientation,** through the same path a NIfTI takes, so one CT read from
     DICOM and the same CT converted to NIfTI arrive identically oriented
 - `describe_series()` checks a series without reading a pixel, because checking a hundred
   cases should not cost a hundred gigabytes
 - Slice spacing is measured from the positions, never from `SliceThickness` - thickness is
   how thick a slice is, not how far apart they sit, and for an overlapping reconstruction
   using it puts every slice past the first in the wrong place
 - A gap in the positions is an error. A volume stacked over a gap does not lose a slice, it
   puts everything past the gap somewhere else, and no metric would show it. With too few
   slices for the median gap to mean anything the message says it cannot tell which gap is
   wrong instead of naming one by arithmetic accident
 - Series reading is wired into the pipeline two ways: a 3D model with several DICOM files in
   one sample, and a path to a series directory in a CSV

### Changed

 - Several volumes per sample - one per modality, as BraTS ships - is refused with a message
   saying so, rather than the first one being read and three quarters of the data ignored

## [0.3.0a1]

*2026-09-26*

Volumes. The spec, the model builder, the losses, the metrics and the tiling were already
rank-generic - `input_shape = [64, 64, 32]` has validated and built a 3D U-Net since
0.2.0a1 - so what was missing was reading volumes, naming classes the way volumes do, and
two refusals that existed to stop a half-working 3D run.

### Added

 - NIfTI reading (`read_volume`, `volume_spacing`), dispatched from `read_image` so
   everything reads "whatever is at this path, the way training would". Two corrections
   happen before anything looks at the array:
   - **canonical orientation.** A NIfTI stores an affine, not an array in a known order,
     and the same anatomy can arrive in any of 48 permutations and flips. Read two datasets
     naively and one has the patient's left on the right; train on both and the model can
     learn which dataset a scan came from. Every volume is reoriented to closest-canonical
     RAS, and there is a test that two files describing the same anatomy through different
     affines come back byte-identical
   - **a fixed intensity window,** the same named windows the DICOM reader uses, so 'lung'
     means one thing across the package. `auto` falls back to the range present for NIfTI,
     which stores no window - stated rather than silent, since that is the one case where a
     bright outlier still rescales everything else
 - `labels` in the data section: masks as label maps rather than pictures, which is how
   every volume format stores them. Exactly one of `colormap` and `labels` - not both even
   when they agree, since two sources for the number of classes means one going stale, and
   not neither, since guessing it from the data would let a validation set with no tumour
   quietly train a model with one class fewer
 - 3D patch sampling, which is `splits` at rank 3 and the same code as the 2D grid
 - `resize_volume()`, through torch rather than scipy: one library fewer to conflict with

### Changed

 - The data pipeline and the engine no longer refuse a 3D spec. Trained end to end on
   synthetic 64x64x32 CT volumes on a GTX 1070: 0.5s an epoch, per-case Dice, and
   predictions returned as whole volumes
 - `dicom_window` is now `window`, since NIfTI needs the same thing and the old name had
   become a lie. `dicom_window` is still accepted - a released R package sends it - and
   giving both is an error rather than a coin toss
 - A class-count mismatch names whichever of `colormap` or `labels` is in use, instead of
   always blaming the colormap
 - `nibabel` is a hard dependency. A medical imaging package that cannot open a `.nii.gz`
   is not one

### Not done, on purpose

 - **Resampling to isotropic spacing.** Spacing is read and carried, because losing it
   would make resampling impossible later; applying it changes the voxel grid the model
   sees, which is a decision to take explicitly rather than inside a reader
 - **Augmentation in 3D.** albumentations ships 3D transforms through a different call
   signature; a 3D spec asking for augmentation is refused, not silently ignored
 - **DICOM series to volume.** Reading one DICOM slice works; assembling a directory of
   them into a volume needs sorting by geometry and a consistency check, and is its own
   piece of work

## [0.2.0a3]

*2026-09-26*

### Added

 - `split_dataset()` - split one directory into train/validation/test CSVs, keeping every
   group whole. `group_by` is a regular expression read against the sample key, and a
   group is whatever must not straddle the split: usually a patient, sometimes a study, a
   scanner or a site. Slices of one patient on both sides of a split is the most common
   way to report a segmentation score that means nothing, and it leaves no trace in the
   output. Deterministic given the same samples, fractions, pattern and seed, in any
   order; groups are handed to whichever split is furthest below its target share in
   samples, so patients of wildly different sizes do not skew the fractions
 - `evaluate_cases()` - one row per case rather than one number per split, optionally
   grouped by patient, and `summarise_cases()` for the distribution: mean, sd, median,
   range, and the name of the worst case. A model averaging Dice 0.855 on the Data Science
   Bowl scored 0.006 on three images; nothing in the mean could say so
 - A `key` column in a `config_file` CSV now names the sample, instead of it being called
   'row 55'. Written by the splitter, so a bad score can be traced back to an image

### Changed

 - Each metric is now one formula, written on overlap statistics and reused for tensors -
   which is what lets a tiled case be scored as a whole. Tiles are summed, not averaged: a
   ratio of sums is not the mean of ratios, and averaging punishes a case whose object
   happens to straddle a tile boundary
 - Finding samples no longer needs a full specification (`discover_samples`), since
   splitting a dataset has no model, no colormap and no loss to invent

## [0.2.0a2]

*2026-09-26*

### Added

 - DICOM reading, done the way the modality requires rather than the way the file
   allows: stored values passed through the modality LUT so they are real units
   (Hounsfield units for CT), a **fixed** window rather than one derived from the image
   itself, and MONOCHROME1 inverted. A window taken per image lets one bright pixel - an
   implant, a marker, an artefact - rescale everything else, and makes two scans
   incomparable
 - `dicom_window` on the data section: a named window such as `lung` or `soft_tissue`,
   an explicit `(centre, width)` pair, `auto` to use the window recorded in the file,
   or `full` for the whole range present
 - Format detected from content, not from the file extension: DICOM files often carry
   none

## [0.2.0a1]

*2026-09-25*

The 2022 TensorFlow package replaced by a PyTorch one. Not a port - a rewrite, sharing
the earlier version's shape (a specification, many models from one file) and almost none
of its code. Everything below is 2D semantic segmentation; the spec and the model builder
already handle volumes, the data pipeline is where 3D stops.

### Added

 - A specification with two constructors and one result: built from arguments or loaded
   from YAML, producing the same validated object, so nothing downstream can tell which
   was used. pydantic v2, `extra="forbid"`, exportable as JSON Schema
 - Four architectures from one builder: U-Net, U-Net++, Res-U-Net, LinkNet, each
   composable with separable convolutions, spatial dropout, learned or interpolated
   upsampling, deep supervision and configurable block width
 - Nine losses (IoU, Dice, CCE, CCE-Dice, Focal, Tversky, Focal-Tversky, Combo, Lovász)
   and three metrics, reducing over every axis except batch and channel - so they work
   unchanged on volumes
 - The engine: trains every model in a specification, then evaluates and predicts,
   reporting a comparison table that names the objective beside the loss column
 - Data pipeline with two layouts, `nested_dirs` and `config_file`, and albumentations
   declared per model
 - Tiling that goes both ways: a large image cut into a grid instead of shrunk, and a
   full-size mask reassembled from the pieces
 - Packaging and CI: published to PyPI through trusted publishing, tested on Ubuntu,
   macOS and Windows across Python 3.10-3.13
 - A `pascal` extra pinning the last CUDA 12 torch, because CUDA 13 dropped the Maxwell,
   Pascal and Volta generations and no driver update brings them back

### Removed

 - TensorFlow and Keras, and with them the `.hdf5` weights format
 - Stacked ensembling and the YOLOv3 branch work: out of scope for v0.1, not abandoned

## [0.1.0rc2]

*2022-10-07*

The TensorFlow package. Still installable as `pyplatypus==0.1.0rc2`, and unrelated to
everything above in implementation.

### Added

 - Platypus engine - allows to train and evaluate multiple CV models with YAML config
 - Generalized multiclass semantic segmentation models: U-Net, U-Net++, Res-U-Net, LinkNet
 - Semantic segmentation loss functions and metrics: Focal loss, CCE, Dice loss/coefficient, Tversky loss/coefficient, IOU loss/coefficient, Focal-Tversky loss, Combo loss, Lovasz loss
 - Semantic segmentation augmentations
