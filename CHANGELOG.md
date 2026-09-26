# Change Log
All notable changes to this project will be documented in this file.

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
