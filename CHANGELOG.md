# Change Log
All notable changes to this project will be documented in this file.

## [Unreleased]

Object detection, end to end: a specification with `task: detection` trains a YOLOv3 through
the same engine as a segmentation one, scores it per class against `pycocotools`-agreeing
average precision, and hands the boxes back in each image's own pixels. On BCCD over five
seeds: `mAP@0.5` 0.8660 +/- 0.0159, `mAP@[.50:.95]` 0.5159 +/- 0.0199, mean overlap of
matched boxes 0.8041 +/- 0.0026 - which brackets and exceeds what a hand-rolled version of
the same recipe scored before any of this existed.

**Still unreleased, and for the same reason as before rather than a new one.** The §4f loop
exists so the R package can reach a feature; R has no detection surface yet, so a release
would serve nothing but the ceremony. It happens when the R side needs it.

### Added

 - `pyplatypus.detection`, starting with **metrics**. The package this replaces had a YOLOv3
   that trained and drew pictures and reported no precision, no recall and no average
   precision - its only number was mean IoU per grid, computed inside the loss, which measures
   agreement with the model's own training target. So there was no way to say whether it
   detected anything, and no number to put beside published weights. Everything else is
   unmeasurable until this exists, which is why it is first
 - `iou_matrix`, `match_detections`, `average_precision`, `detection_report`. Pure numpy and
   no torch: they work on boxes, so they are testable without a model
 - **Agreement with `pycocotools` is a test, not a one-off check.** mAP@0.5 and mAP@[.50:.95]
   match the COCO reference implementation to 1e-12 - the tolerance the suite asserts - on
   five reproducible problems, and those
   values are baked into the suite so the agreement is verified on every run without
   pycocotools installed. A detection number that disagrees with COCO is a number nobody can
   compare to anything

### Added — detection in the specification

 - **`task`**, a discriminator on the specification: `segmentation` (the default) or
   `detection`. It selects the *type* of `data` and of every entry in `models`, rather than
   switching a flag, so a detection configuration carrying a `colormap` or a segmentation one
   carrying `anchors` is refused by name instead of silently ignored. One task per spec, not
   one per model, for the same reason there is one rank per spec: every model in a spec shares
   a data pipeline, and masks and boxes are not the same pipeline
 - A configuration with no `task` is a segmentation one, so every file and every released R
   package written before this keeps working unchanged
 - `DetectionData` - `classes` in class order (position is the class index, stated rather than
   read from the files, where it would come out alphabetical), `annotation_format`,
   `coordinates` for the VOC 1-based-inclusive convention, `strict_labels`. No background
   entry, unlike a colormap: a region of an image holding no object is not a box
 - `DetectionModel` - `anchors` as fractions of the input, fitted to the training data when
   unset; `anchors_per_grid`, `ignore_threshold`, `score_threshold`, `nms_threshold`,
   `operating_point`. Every one of these was a command-line flag in
   `examples/detect_blood_cells.py`, which is where a setting lives when it has nowhere better
   to be
 - `DataSpec` and `ModelSpec`, the shared bases the two tasks are built from. What they
   deliberately do **not** carry is `loss` and `metrics`: segmentation picks both from a menu
   of nine and three, YOLOv3's objective is part of its architecture, and mean average
   precision is not one option among several. A field that accepts a value and ignores it is
   worse than no field
 - Refusals that cost nothing now and a confusing failure later: detection at rank 3 (ImageNet
   is images and YOLOv3 is a 2D architecture), an input side that is not a multiple of 32,
   anchors given in pixels rather than fractions - with the conversion named, since COCO
   publishes its nine in pixels at 416 - anchors that disagree with `anchors_per_grid`, grids
   with different anchor counts, duplicate class names, and `coordinates` set for LabelMe,
   which stores continuous pixel coordinates and has no convention to pick

### Added — detection in the engine

 - **`DetectionEngine`**, and `build_engine(spec)` which returns it or `Engine` according to
   the task, so a caller holding a spec never has to ask. Two classes rather than one with a
   branch in every method: past discovery the two tasks differ at every step - three target
   tensors instead of one, a loss that is part of the architecture, and an evaluation that
   cannot be accumulated per batch because average precision is a property of a whole split's
   ranking
 - `DetectionDataset` - one sample to one example, through the letterbox and the target
   encoder. One sample is one example, with no tiling: cutting an image into a grid cuts boxes
   in half, and half a box is not a smaller object
 - `DetectionTrainer`, reporting the four parts of the loss separately for train and
   validation. A YOLOv3 total means nothing alone - cross-entropy against a soft target bottoms
   out at the target's own entropy, so the coordinate term has a floor above zero that depends
   on the data. `coordinates 5.93 / objectness 0.001` is the difference between a model that has
   learned what is where and one that has stalled, and the total cannot tell them apart
 - **Anchors are fitted to the training annotations** when the spec does not give them, and
   recorded on the run and in the weights sidecar. Not optional metadata: the same weights read
   with other anchors decode every box scaled by a fixed factor, with plausible boxes, plausible
   scores, wrong places, and nothing in any output to say so
 - **The targets are surveyed before the first epoch**, and a dataset the encoder loses more
   than 2% of warns. Two boxes of one shape whose centres land in one cell share a slot, so the
   second is never shown to the model and never counted as missed. Measured on BCCD at a 416
   input: 3 boxes of 2804
 - `evaluate()` returns the detection columns - `map_50`, `map_50_95`, `mean_matched_iou` - and
   **no overall precision or recall**. Averaging those over classes needs a weighting and every
   weighting is a different claim: summed over BCCD's 4155 red cells, 372 white and 361
   platelets, a single precision is a statement about red cells wearing the costume of a
   statement about the model. They are in `evaluate_classes()`, per class
 - `predict()` returns a **list, in each image's own pixels**. There is no array form - images
   differ in size and so does the number of boxes found in each - and a box in a letterboxed
   416x416 frame cannot be drawn on the photograph it came from. The same decision as
   `space="source"` on the segmentation side, except that here there is no alternative
 - `anchor_coverage()`, asked of whichever split you like. The useful question is about a split
   the anchors were *not* fitted on: 0.92 on training and 0.70 on validation says the two hold
   different objects, and no training curve shows that
 - A **test split without annotations** can be predicted and refuses to be scored. Scoring it
   against empty truth would report zero, which is a different statement from "there is nothing
   to score against"
 - `config_file` mode reads an **`annotations` column** for a detection spec - the second slot
   is named after what labels an image, and `write_splits(label_column=...)` writes it

### Added — a learning-rate schedule the specification can state

 - **`cosine_annealing`**, a callback decaying the rate from its initial value to `min_lr`
   along a cosine. No `monitor`: it is a function of how far through the run you are rather
   than of how the run is going, which is the difference from `reduce_lr_on_plateau`, and a
   run may legitimately want both.

   It exists because wiring detection into the engine found that **the specification could
   not express the run that produced this package's published detection number.** The
   measured BCCD run decayed its rate; a spec could only hold it constant.

   What the schedule is actually worth was then measured, and it is less than the gap it
   was reached for: adding it moved BCCD's test `mAP@[.50:.95]` from 0.4700 to 0.4761 and
   `mAP@0.5` from 0.8479 to 0.8496. **The first explanation of that gap was wrong**, and
   the measurement refused it; the rest of it was augmentation, which the engine did not do
   at all - the entry below. Attributing a gap to the plausible cause is the mistake this
   project keeps catching, and this time it took two tries.

   Both of those runs are single draws, so part of +0.0061 is the schedule and part is the
   draw - and five seeds of the finished recipe later put the standard deviation of
   `mAP@[.50:.95]` at 0.0199, which is three times that difference. The schedule is kept
   because it is the recipe the measurement used and because the spec could not state it,
   not because its effect was established. `DETECTION_RECON.md` §12 has the account.
 - Checked against `torch.optim.lr_scheduler.CosineAnnealingLR` epoch for epoch, at four run
   lengths and two floors. "The rate goes down" is not the claim worth testing: if this were
   a schedule that merely resembled torch's, comparing a run through the specification with
   the measured one would compare two experiments and blame the difference on the wrong
   thing.
 - Each parameter group decays **from its own initial rate**, so `encoder_learning_rate` is
   not flattened at the first epoch - which would have been invisible, since the history
   records one `learning_rate`, the first group's.
 - `TrainingState` gained `total_epochs`. A schedule that is a function of progress needs
   the horizon, and nothing else in the state implies it: `epoch` counts up and stops
   wherever early stopping stops it.

### Added — augmentation for boxes

 - **`BoxAugmenter`** and `build_box_augmenter`: a spec's `augmentation` list now applies to
   a detection run, with the boxes following the pixels. albumentations again, declared the
   same way as for masks, so a detection spec and a segmentation spec name transforms
   identically
 - **Every geometric transform in albumentations 2.0.8 moves boxes with the pixels, which was
   measured rather than assumed.** A bright rectangle with the box labelling it, twenty seeded
   draws per transform, comparing the box that came back with the bounding box of the bright
   pixels in the output. Eight are exact to 0.00 px - the flips, `RandomRotate90`,
   `Transpose`, `D4`, a pure translation, the crops - and five land within 1.00 px: `Affine`
   with a rotation or a scale, `Rotate`, `Perspective`, `GridDistortion`. For those five the
   box bounds the warped corners while the pixels are what survived interpolation, so the
   test asserts the box still **covers** the object rather than equals its extent. A property
   of the library, so it lives in the tests where a version bump re-runs it
 - `min_visibility` on a detection model: how much of a box must survive a transform that
   removes part of the frame. A convention rather than a measurement, and the direction is
   what is defensible - a box keeping two pixels of a cell teaches the model that a two-pixel
   fragment is a whole cell, which produces false positives everywhere, while dropping a
   heavily truncated object only fails to teach it about that object. Pascal VOC marks such
   objects `truncated` for the same reason. Irrelevant unless a transform can lose part of
   the frame, which flips and rotations never do
 - Training is augmented and validation never is: measuring a model on distorted data
   measures the distortion

### Fixed

 - **A probe that failed for its own reasons.** The check that names a transform which cannot
   handle boxes ran against a fixed 32x32 image, so `RandomCrop(height=64, width=64)` - a
   perfectly good transform - was refused with "crop size exceeds image dimensions". A probe
   that fails for the wrong reason is indistinguishable from a real refusal, and this one
   reported the user's transform as unsupported when it was the check that was too small.
   Both probes now run at the model's own input size, which also keeps the refusal correct
   for a crop genuinely larger than the input.

   **The 3D volume probe had the same defect** and had had it since it was written: a fixed
   4x8x8 volume refused any larger 3D crop. Found only because the box probe made the shape
   of the mistake obvious.
 - `albumentations` rejects bad parameters with a pydantic `ValueError`, not Python's
   `TypeError`, so a misspelled argument escaped the handler that exists to name it and
   arrived as schema prose. Both are caught now
 - **A decaying learning rate was invisible in the output.** `learning_rate` was printed to
   four decimal places like every other quantity, so a cosine going from 1e-4 to 1e-8 read
   `learning_rate=0.0000` from epoch 90 on, and a run with a schedule looked identical to one
   without. Found by watching a run and trying to confirm the rate was moving; no test would
   have asked
 - **`seed` did nothing.** The field has been on the specification since the first release,
   described as "set it if you want a reproducible run", and no code read it - so in R,
   `platypus_spec(seed = 1)` made a promise and two runs of it disagreed. Both engines now seed
   `random`, numpy and torch at construction, which also seeds the generator `DataLoader`
   derives its workers' seeds from, so augmentation in a worker process is reproducible too.
   Found while wiring detection in, where the anchors are fitted by k-means and the seed had to
   come from somewhere.

   Asserted by training twice and comparing, and - the half that makes the first half mean
   anything - by checking that two *unseeded* runs differ. Without that second test the first
   would pass on a package that ignores `seed` entirely, which is the state it was in.

   What it does not do is ask torch for deterministic algorithms: that makes some convolutions
   much slower and makes others raise, so the result would be a seed that sometimes refuses to
   run. Two runs at one seed on one machine agree; across machines or cuDNN versions they need
   not

### Changed

 - **`examples/detect_blood_cells.py` was rewritten onto the specification**, which is the test
   of whether the specification earned its place. The first version assembled a detector by
   hand out of `pyplatypus.detection`: a Dataset class, a training loop, a decode-and-score
   function and twenty flags. What is left is the part that is genuinely about BCCD - where its
   files are and what its canonical splits say - and the script writes out the YAML of the run
   it just did, beside the results
 - **Loading weights into a detector adopts the anchors recorded beside the file**, because a
   detector's weights only mean anything with them: read with any others, the same weights
   decode every box scaled by a fixed factor - plausible boxes, plausible scores, wrong places,
   and nothing in the output to say so. A file that carries no anchors is refused rather than
   guessed at, and so is one whose anchors-per-grid does not fit the head.

   A specification naming both `weights` and `anchors` is **refused**: two claims about one
   model with one of them untrue. Taking the file's and warning would mean the specification
   no longer describes the run, which is the property everything else here rests on.

   The engine also checks what the model specification cannot answer for itself - `n_class`,
   which sets the head's width and lives on the data, and the class **names**, because weights
   trained on the same number of differently ordered classes load cleanly and label every box
   wrongly. The detection counterpart of weights trained on a different colormap.

   `fit: false` is therefore a usable thing for a detector: load a published detector and
   predict with it, which is what a vignette needs to open with.

   The test is a round trip - train, export, load into a specification that names no anchors at
   all, and compare the **boxes**. Verified to catch wrong anchors by scaling the adopted ones
   by 1.2
 - `weights_fingerprint()` moved onto the spec, so what identifies a file of weights is an
   architecture's own answer. `blocks` and `filters` identify a U-shaped model and mean nothing
   to a detector, whose head width is set by its anchor count instead - a fixed list in
   `pyplatypus.weights` had to reach for a field that is not there
 - A union tag no longer reads as a field. pydantic puts the matched tag at the head of every
   nested location, so a problem in a detection spec was reported at
   `detection.models[0].anchors`; it now says `models[0].anchors`. An enum used as a tag is
   also no longer rendered with `repr`, so a mistyped task is told to use `'detection'` rather
   than `<Task.DETECTION: 'detection'>`
 - The JSON Schema's top level is now a `oneOf` over the two tasks with a `discriminator`, and
   the per-task fields live under `$defs`. Its docstring claimed it was shipped inside the R
   package; it is not, and has not needed to be - the R package sends the configuration to the
   engine and translates the engine's refusal, which keeps one validator instead of two that
   can disagree
 - `Engine` refuses a detection spec and says why. The spec validates and nothing trains from
   it yet; saying so beats failing somewhere inside the data pipeline, and beats a feature that
   is advertised and does not work. A detector is still assembled by hand from
   `pyplatypus.detection`, the way `examples/detect_blood_cells.py` does
 - `PlatypusSpec` is now the shared base of `SegmentationSpec` and `DetectionSpec` and is not
   built directly. `from_dict` and `from_yaml` return whichever the task asks for, and both
   are instances of it, so nothing that only needed a seed or a rank changes. Building the base
   directly says so rather than failing as two "extra inputs are not permitted"

### Added — annotations, letterboxing and the target encoder

 - `read_voc`, `read_labelme`, `read_annotations` and `describe_annotations`. Pascal VOC
   first, because that is what BCCD ships and what the old package's readers handled
 - `Letterbox`, which scales by one factor and pads, **replacing the plain resize the old
   package used**. That resize stretched a 640x480 photograph; it was internally consistent,
   since boxes were normalised by the source dimensions, but the COCO weights were trained
   on letterboxed input so loading them onto stretched images is a silent mismatch. The
   transform is reversible and carries what it did, so a box predicted in the network's
   space goes back onto the pixels it came from - the detection counterpart of
   `predict(space="source")`. Padding is grey rather than black: black is a legitimate pixel
   value in a radiograph
 - `encode` / `decode`, boxes to the three target tensors and back, **tested as exact
   inverses**. `anchors_per_grid` comes from the anchors given rather than a constant, and
   anchors are fractions of the input so the same ones describe the same shapes at 416 and
   at 608
 - `Encoding.unplaced` counts the boxes that **cannot be represented**: two objects of one
   shape whose centres fall in the same cell share one slot, so the second is lost. At the
   density of a blood smear a 416 input loses about **one object in eleven** and a 608 input
   loses none, which makes the input size a decision rather than a default - and is the
   difference between "the model is poor" and "the target never held half the cells"

### Added — a detector, and the first number anyone can compare

 - `Yolo3`, `Darknet53` and `build_yolo3`. **61,949,149 parameters for 80 classes and three
   anchors, which is YOLOv3's published figure** - the one external check on the
   architecture's inventory, and what the COCO weights require, since a structure that
   merely resembles Darknet-53 loads them cleanly and predicts nonsense
 - `Yolo3Loss`, reported in four parts because one number cannot say what is wrong: a run
   whose coordinate loss falls while its objectness does not is finding the right places and
   refusing to commit. The ignore mask is the part most reimplementations drop - a cell that
   was not assigned a truth but predicts a box overlapping one contributes **nothing** to
   the objectness term, rather than being trained towards zero, which would teach the model
   to suppress correct answers
 - `non_max_suppression`, **per class by default**: a platelet on a red cell is two objects
   at one place and suppressing across classes deletes one of them
 - `DetectionMetrics.mean_matched_iou` and a third return from `match_detections`: how well
   the boxes that matched actually fit. Average precision uses IoU as a *threshold* and says
   nothing about the fit, so this is the quantity the AP figures only imply. Weighted by how
   many matched, not a mean of per-class means
 - `examples/detect_blood_cells.py`: BCCD from `github.com/Shenggan/BCCD_Dataset` (MIT, and
   not the Kaggle mirror, for the reason given in 0.3.0a8's notes), trained from nothing,
   with its own train/val/test split

### Measured

150 epochs, 33 minutes on a GTX 1070, BCCD's own test split:

    mAP@0.5 0.8576    mAP@[.50:.95] 0.5050    mean IoU of matched boxes 0.8046

    class       AP@0.5    IoU   truth   prec@0.5   rec@0.5
    RBC         0.7986  0.809     805      0.720     0.804
    WBC         0.9546  0.854      71      0.922     1.000
    Platelets   0.8197  0.707      69      0.524     0.942

The first number in this package's history that can be compared with anybody else's - the
old YOLOv3 reported mean IoU per grid from inside its loss and nothing else, which is why
its Blood Cell Detection example ended on a picture.

**The ceiling is localisation, and it is size-dependent exactly as geometry predicts.** AP
by threshold: white cells hold 0.789 at IoU 0.80 while platelets fall from 0.820 to 0.041.
A two-pixel error per side costs a 30-pixel box a sixth of its overlap and a 200-pixel box
almost nothing. So mAP@[.50:.95] on a dataset of small objects measures how finely a box can
be placed, not how well objects are found.

**And the loss converged rather than stalled.** It plateaus at 5.936 against a floor of
**5.891**, computed exactly on real batches: cross-entropy against a soft target bottoms out
at the target's own entropy, so the coordinate term cannot reach zero. Two runs agreed to
0.003 of mAP.

### Added — anchors fitted to your own boxes

 - `generate_anchors`, `fit_shapes`, `box_shapes` and `anchor_coverage`. k-means over box
   shapes with **IoU as the distance**, k-means++ seeding, carried over from the old
   package - the piece of it with no equivalent anywhere in R
 - `AnchorFit` carries `mean_iou`, the quantity the fit maximises and the number that says
   whether these anchors describe this data. Comparing it across `anchors_per_grid` is how
   the count gets chosen instead of assumed; the old implementation **printed a class table
   and drew a scatter plot as side effects and reported no number at all**
 - `anchor_coverage` answers "are COCO's anchors good enough for my data" in one number
   rather than a training run. On blood-cell-shaped boxes: **0.67 borrowed against 0.92
   fitted**
 - **The whole pipeline is verified against COCO's published anchors.** Plant the nine as
   clusters of boxes, refit from nothing, and COCO's anchors come back in COCO's groups to
   within 1-3 pixels at 416 - which checks the seeding, the distance, the convergence and
   the grouping at once

### Fixed, in work carried over rather than in released code

 - **Anchors are grouped into scales by area, not by width.** Verified against the one
   external reference there is: sorting the nine COCO anchors by area reproduces their
   published grouping exactly, and sorting by width - what the old code did - does not. It
   puts (30, 61) in the finest grid and (33, 23) in the middle one, swapping them
 - **Box shapes are measured in the space the encoder uses.** The old code divided a box by
   its source image, which equals the fraction of the network input only when the image is
   stretched to fill it. With letterboxing a square object in a 640x480 image is square
   after letterboxing and **33% taller than it is wide** under source normalisation. For a
   dataset of one image size the distortion is uniform and the model compensates; it stops
   being harmless the moment sizes are mixed or COCO weights are loaded

 - **Pascal VOC's one-pixel convention.** VOC stores 1-based inclusive indices, so pixels
   1 to 10 are ten pixels wide and `xmax - xmin` is nine - leaving **81% of the true area**,
   which on a fifteen-pixel platelet is not a rounding question. The minima have 1
   subtracted, once, where the format is known. Choosing wrongly is caught rather than
   silent: a minimum of 0 cannot occur under a 1-based reading, so a file containing one is
   refused with the alternative named
 - **An infinity in the old target encoding.** It stored `logit(centre - floor(centre))`,
   and that fraction is zero whenever a box's centre lands on a cell boundary. Measured on a
   416 input at grid 13: **37 of 1200** integer-pixel boxes, so about one in thirty had a
   `-inf` target. The target now holds the offset itself and the loss applies the sigmoid,
   so nothing is inverted and nothing can be infinite

### Decided, and said in the code rather than assumed

Two numbers both honestly called "mAP" can differ by points on identical predictions, so
`detection_report` returns the conventions it used beside the score.

 - **Boxes are continuous `(xmin, ymin, xmax, ymax)`.** Pascal VOC stores pixel indices and
   adds 1 per side, which raises a 10-pixel box's IoU by 21%; that belongs in the annotation
   reader, not here
 - **Matching is greedy by descending score, one truth each**, so a duplicate box is a false
   positive - which is what makes non-maximum suppression worth doing
 - **Predictions are pooled across images before AP**, never averaged per image. The curve is
   one ranking over the dataset; a mean of per-image APs is dominated by images with one
   object. Same reason the segmentation metrics sum TP, FP and FN before applying their
   formula - a ratio of sums is not the mean of ratios
 - **A class with no ground truth scores `None`, not zero**, and is left out of the mean;
   `classes_without_truth` says which. Three interpolations are offered - `all` (VOC 2010+),
   `101` (COCO) and `11` (VOC 2007) - because papers quote all three
 - `difficult` objects are excluded when annotations carry the flag, as VOC's own evaluation
   excludes them

## [0.3.0a11]

*2026-10-03*

### Added

 - **`fit()` refuses to train on masks the colormap does not describe.** A colormap matching
   none of the labelled tissue is the quietest failure in segmentation: every mask reads as
   background, the loss falls because background is most of a medical image, the metrics look
   plausible, and the model learns to answer "nothing here". Nothing in a training log says so
 - The refusal is on **class presence, not on the unmatched fraction**, and that choice is the
   substance of this change. The unmatched fraction scales with the size of the thing being
   segmented, so on a small lesion - the ordinary medical case - the most dangerous mistake
   makes the quietest signal. Measured on synthetic masks with a colormap asking for VOC red
   where the masks are white: a foreground covering 20% of the image gives 19.8% unmatched,
   5% gives 4.96%, and **0.6% gives 0.61% - indistinguishable from the 0.72% that JPEG
   compression leaves around the edge of a correct mask.** A declared class that appears in no
   mask does not scale: the same measurement gave per-class pixel counts of [15984, 400] for
   the correct colormap and [16384, 0] for the wrong one, at any lesion size
 - `n_class` higher than the data has classes is caught by the same check and is invisible to
   the fraction entirely - every voxel matches an entry, so the unmatched figure is exactly
   zero while one output channel can never be trained
 - `SegmentationDataset.inspect_masks()` returns a `MaskReport`: the unmatched fraction, which
   classes appear, which are missing, and how much of the dataset was read.
   `colormap_coverage()` is unchanged and now delegates to it
 - A high unmatched fraction with every class present is a **warning** rather than a refusal,
   because some of it is legitimate. The threshold is 5%, which sits in the measured gap
   between compression artefacts (under 1%) and a wholesale mistake (19.8% and 100%)
 - `Engine(..., check_masks=False)` skips it, and the refusal names that along with the call
   that looks at more of the data than the check does. Both were run as written before being
   printed

### Changed

 - The sample the check reads is **spread across the dataset and ordered so that any prefix
   still spans it** - first, last, middle, then midpoints. Real datasets arrive sorted, by
   patient or acquisition date or class, so a prefix is a biased sample and "this class is
   absent" drawn from one can mean only "the positives are later in the list". Tested on a
   dataset whose only labelled tissue is in the last 3% of a sorted listing, which a prefix
   would have refused
 - The scan **stops once every class has turned up** and at least five masks have been read,
   so effort goes where there is doubt. On Data Science Bowl, where each sample carries one
   mask file per nucleus, a fifty-sample scan cost 3.8 seconds; it now answers in 0.13

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
