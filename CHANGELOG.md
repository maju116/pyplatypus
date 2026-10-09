# Change Log
All notable changes to this project will be documented in this file.

## [0.8.0a4] - 2026-10-09

### Added

 - **`examples/segment_retinal_vessels.py`** - FIVES retinal vessels, and the first example
   that sets `splits`. It exists because three things had no worked demonstration: tiling, a
   whole-mask metric, and scores reported per group rather than as one mean.

   A FIVES fundus is 2048 x 2048 and its vessels are about **nine pixels** wide, measured as
   mask area over skeleton length. Resized to 256 they are 1.2, and clDice reads 0 for a
   one-pixel structure displaced by one pixel - so the usual resize lands the data where no
   overlap metric can tell a near miss from a total one. That is the argument for `splits`,
   end to end.

   It scores twice on purpose. Dice and IoU come from the tiled run; clDice cannot, because a
   skeleton is a property of a whole mask and a tile severs every vessel crossing its edge -
   the specification refuses the combination. So a second pass steps the model over one
   retina's tiles, stitches, and scores that. One at a time rather than through `predict`,
   which holds three copies of a split and was killed on this one (pyplatypus#178); streaming
   it peaks at 2.22 GB, and the skeleton on the card takes the pass from 11.96 s an image to
   0.37.

   The run, 40 epochs over 600 retinas, 403.4 minutes, 1,942,594 parameters:

       dice 0.8509   iou 0.7739          (best at epoch 34; nothing after it improved)

       per disease          dice                 clDice, whole images
       AMD             0.9105 +-0.0474        0.9137 +-0.0530
       diabetic        0.8792 +-0.0697        0.8886 +-0.0652
       glaucoma        0.8393 +-0.1770        0.8402 +-0.1771
       normal          0.9020 +-0.0452        0.9051 +-0.0500

   **The glaucoma spread is not glaucoma**, and the per-disease table is in the example
   because finding that out needed it. It replicated across two independent runs, so it is
   the data; FIVES then ships a quality grading that explains it. Fourteen of the fifteen
   low-contrast test images are glaucoma, and on the images FIVES grades clean, glaucoma is
   the *best* of the four and the tightest - 0.9325 +-0.0232. clDice splits the same way, so
   it is not a Dice artefact.

   And **Dice and clDice agree here** - correlation 0.9779, mean difference +0.0041 at a
   spread of 0.0219 over 200 cases. Said plainly because a new metric that reorders nothing
   should be reported as redundant at this quality level rather than as a second opinion. What
   clDice adds on this run is the three worst cases, not the ranking of the four diseases.

## [0.8.0a3] - 2026-10-08

### Changed

 - **A tiled run shuffles in windows, and trains roughly twice as fast.** The dataset has
   always cached decoded samples - `_load` says *"cached because every tile asks again"* -
   but training shuffled over **tile** indices, so a cache of eight held nothing against a
   working set of six hundred and nearly every tile paid for its own decode.

   `TileShuffle` draws a window of samples, shuffles all of their tiles together, and moves
   on. One decode then serves every tile of its sample, and a batch still draws from several
   sources. The window is the cache size by construction: a wider one would evict a sample
   while its own tiles were still being asked for.

   Measured on the loader alone, FIVES retinas at 2048 cut sixteen ways:

       96 tiles in order      4.68 s     49 ms a tile
       96 tiles shuffled     24.37 s    254 ms a tile
       96 tiles, windowed     4.72 s     49 ms a tile

   And end to end, Data Science Bowl cut 2x2 at 128, three seeds of thirty epochs each:

       windowed       dice 0.8996 +/-0.0024    23.0 s an epoch
       full shuffle   dice 0.9036 +/-0.0037    40.6 s an epoch

   **The speed is established and the quality difference is not.** Paired by seed the
   difference is +0.0023, -0.0060, -0.0082 - mean -0.0040 against a standard error of
   0.0032, so 1.24 standard errors, with two seeds of three favouring full shuffling and one
   against. A hint rather than a finding, and three seeds cannot settle a difference that
   size. Recorded here rather than rounded away: at 1.76x the epochs for the same minutes, a
   run can buy back more than 0.004 of Dice, which is what makes shipping it the honest call
   rather than a convenient one.

   Nothing external changes, and no interface moved. A seeded tiled run will not reproduce a
   result from before this release, because the order the tiles arrive in is different -
   which affects nobody known: **no example, configuration or article sets `splits`**,
   checked rather than assumed.

 - `SegmentationDataset.cache_size` is public, because the loader sizes its window from it.

## [0.8.0a2] - 2026-10-08

### Added

 - **`cldice`, a metric for structures where Dice asks the wrong question.** Centerline
   Dice (Shit et al., CVPR 2021): each mask is scored against the other's skeleton, so it
   measures whether a structure is *connected* rather than whether its pixels coincide.
   Vessels, airways, neurons, cracks, roads, catheters.

   The argument, measured here rather than quoted: a nine-pixel vessel drawn two pixels thin
   scores Dice **0.875**, and the same vessel severed in the middle scores Dice **0.947** -
   Dice prefers the severed one. clDice gives them 1.000 and 0.943, which is the other way
   round. Its limit is measured too: at one pixel wide, a one-pixel displacement leaves no
   overlap at all and every overlap-based metric reads 0, this one included.

   The soft skeleton is iterated min and max pooling, which is differentiable, so the same
   function can serve a loss later; a hard skeleton would have been cheaper and not
   comparable with the paper. Rank-generic, and **verified against `scipy.ndimage`'s own
   morphology** - exact agreement, 2D and 3D, on random masks.

   Two silent failures are refused rather than scored:

   - a structure thicker than `iterations` can peel has an **empty** skeleton, and the
     ratios then read smooth/smooth. Measured: a 24-pixel square at `iterations=1` gives
     clDice **1.0** to a prediction whose Dice is 0.0017. Refused by name;
   - `include_background` defaults to **False** here where every other metric defaults to
     True. The background's skeleton lies inside the background by construction, so that
     class scores 1.0 whatever the model did - averaging it in moved 0.7865 to 0.8933 on a
     three-pixel vessel. Not a flattering average: a number with no meaning in it.

### Changed

 - **A tiled run refuses a whole-mask metric.** `splits` hands the model pieces, and a
   skeleton is a property of a whole mask, so every structure would be severed at four tile
   edges and come back low for a model that is perfectly connected - silently. Refused when
   the specification is read, which is before anything is loaded. Dice is unaffected: its
   pieces sum, which is what `combine` exists for.

## [0.8.0a1] - 2026-10-08

### Added

 - **The engine draws.** `plot_masks`, `plot_boxes`, `plot_anchors`, `overlay_mask` and
   `overlay_agreement`, under the names the R package gave them, returning a matplotlib
   `Figure` or an RGB array and never showing or saving anything - the caller decides.
   matplotlib is a **hard dependency**: a plotting function behind an extra is a documented
   feature that fails for whoever did not know to ask for it, which has happened here
   before.

   The decisions a drawing makes - which colours mean found, missed and invented, how much
   of the image an overlay lets through, how a box is labelled, which boxes are worth
   drawing - live in `pyplatypus.style` and are reachable in one call as `drawing_style()`.
   Two packages that each chose their own would come to disagree about what red means.

 - **`read_image` is public**, with the `Args` and `Returns` it never had. Boxes come back
   from `predict` in the source image's own pixels, so drawing them needs the image at its
   own size, and until now that meant reaching into `pyplatypus.data`. R has exported
   `read_images()` since the rewrite began.

 - **`plot_anchors` takes a `DetectionEngine`**, as R's takes a fit, and asks it for the
   anchors and the box shapes in one call. Widths from one place and anchors from another
   is how a figure comes to show boxes in different places from where the anchors were
   fitted to them, which looks like a bad fit and is a bug.

 - **`plot_masks` draws a plane of a volume**, and refuses a volume without one. A volume
   shown as one picture is either a lie or a projection nobody asked for.

 - **`--figures` on all four example scripts**, each drawing what its table cannot say.

### Changed

 - **Nine engine methods take the default they documented.** Every one of them said
   `model_name` "defaults to the first" and every one required it, so omitting the name
   raised `TypeError` on nine reference pages. `evaluate_cases`, `predict`, `report`,
   `crops`, `box_shapes`, `anchor_coverage`, `evaluate_classes` and `evaluate_images`,
   across both engines. The resolution is one method on `EngineBase`, so the two cannot
   come to answer it differently.

 - `plot_masks` defaults to the black-and-white colormap R defaults to, and `plot_anchors`
   to a linear scale as R does.

### Fixed

 - **A mask holding a class the colormap has no colour for was drawn rather than refused**,
   on one of the two ways in. A one-hot mask with too many channels was refused by name; an
   index mask was clipped, so a four-class mask drawn with two colours showed classes 2 and
   3 in class 1's colour and said nothing.

 - **`plot_boxes` labelled every real prediction with an integer.** `predict` puts the
   class indices under `labels` and the names under `names`, and the figure read `labels`.
   Both halves were green and the join was wrong.

 - **Three example scripts could not build a configuration at all**, and had not been able
   to since 0.5.0a1: they sent `n_class`, which left the specification in that release, and
   three of them never sent the required `task`. Nothing ran them - `examples/` is in
   ruff's gate, which reads their layout. `tests/test_examples.py` now hands each one's
   configuration to the loader, with the arguments read off each script's own
   `parser.add_argument` calls so a renamed flag is followed rather than restated.

## [0.7.0a3] - 2026-10-07

### Added

 - **A guide to writing a configuration**, by hand, from one model to several - and a
   **page that explains every field**, generated from `spec_schema()` at
   docs-build time: every block, every field, its type, its default and the description the
   pydantic model carries. 206 descriptions were written for 0.7.0a1 and **until now the only
   way to read one was to call `spec_schema()` in Python** - the site's own "see every field a
   configuration accepts" pointed at that function, which is documentation that requires an
   interpreter. An R user could reach them not at all.

   It is the engine's page and there will not be a second one. The configuration is the one
   thing that is *identical* between the two packages - the same YAML, read by the same models
   - so the R package's site links here rather than keeping a copy, which is the arrangement
   §4aa of the project notes spent a day establishing elsewhere.

 - **A Citation page, and `CITATION.cff`.** The R site had one and this one did not: R
   generates it from DESCRIPTION, which is that language's convention, and CFF is this
   one's - GitHub reads it for "Cite this repository", and the page is rendered from it so
   the two cannot disagree. A test holds its `version` to `pyproject.toml` and
   `__version__`, because it is a committed file no release step reads and nothing else
   would notice it going stale. The year comes from `date-released` rather than the clock,
   so a page built next year does not restate the citation as next year's.

   The About section is one ordered list now rather than pages-then-links, which is what
   forced Licence before Citation and could not express the order the R sidebar uses.

 - **The JSON Schema is published**, so `# yaml-language-server: $schema=...` on the first
   line of a configuration gives completion and checking in any editor using
   `yaml-language-server`.

### Fixed

 - **`SCHEMA_ID` named a URL that was a 404.** It said
   `maju116.github.io/platypus/schema/spec.schema.json` - the *R package's* site - and nothing
   there served it, so the `$id` promised editor validation that could never work. Verified
   404 on both sites rather than assumed. It is now on this package's own site, which is where
   the file is generated and published from; pointing it at the R site would mean one
   repository serving another's artefact for no gain. The old value cannot be in use by
   anybody, because it never resolved.

### Notes on how the page is built, because three choices were measured rather than picked

 - **Definition lists, not tables.** A table was the first shape, and measuring the
   descriptions killed it: median 202 characters, ninth decile 383, **four carry fenced code
   blocks** - which do not render inside a `|` cell - and one carries a `|` of its own, which
   silently splits the cell it lands in.
 - **Unions are named by their `name:` tags**, not by the pydantic classes behind them,
   because the tag is what gets typed. Those discriminators sit on the *field* rather than
   behind a `$ref` - `loss`, `metrics`, `optimizer` and `callbacks` all do - and reading only
   the `$ref` left four fields described as "any".
 - **Every cross-link points at an explicit `{#id}`.** Guessing how the renderer derives an id
   got it wrong: pandoc keeps the underscore in `task: semantic_segmentation`, so a link
   written as `#task-semantic-segmentation` went nowhere. Measured in the rendered HTML.

 - **Every configuration the guide shows is validated by a test**, which is the check a
   written-by-hand page needs and a generated one does not. It earned its place while the
   page was being written: three of the six blocks were wrong before the prose was finished
   - `model_checkpoint` has no `mode`, it does require a `path`, and `anchors: auto` is not
   a thing, `auto` belonging to `window` while anchors are fitted by being left out. A
   reader would have copied each one and been refused. The guide's section titles are pinned
   too, because the R package's guide walks the same ones.

 - Five tests on the generated page, each proven able to fail. Coverage is asserted against the schema's own
   reachable definitions rather than against a count - a count is what let §4as's coverage
   guard pass with a third of the model tree invisible to it - and removing the nested-block
   expansion, which is how `split:` and `augmentation:` reach the page, fails two of them.

## [0.7.0a2] - 2026-10-07

### Fixed

 - **`available_transforms()` decides membership by type rather than by the shape of the
   name**, so the ten things it offered that a specification cannot name are gone: the eight
   composition classes (`Compose`, `OneOf`, `OneOrOther`, `RandomOrder`, `ReplayCompose`,
   `SelectiveChannelTransform`, `Sequential`, `SomeOf`) and the two parameter dataclasses
   (`BboxParams`, `KeypointParams`). None was ever usable - an `AugmentationStep` is a flat
   name and a dict of parameters with no nesting - so naming one passed the check here and
   failed when the pipeline was built, two layers below the question asked.

   The old filter kept anything capitalised that did not start with `Base`, `Basic` or
   `Dual`, which was doing two jobs at once: excluding the interface classes and excluding
   non-transforms. Type does the second properly. For the first there is **no rule to
   derive, and that was measured rather than assumed** - `DualTransform()`,
   `ImageOnlyTransform()` and `Transform3D()` all instantiate without complaint, forty
   genuine transforms inherit `apply` instead of defining it, and "is a superclass of
   another public transform" catches `Affine`, `Blur`, `HorizontalFlip`, `NoOp`, `Pad` and
   `D4`, which are all usable. So those four are excluded by identity, with a test pinning
   the resulting counts so a base added upstream fails rather than appearing silently.
   `NoOp` is why the defining module cannot decide it either: a real transform living beside
   the four in `albumentations.core.transforms_interface`.

 - **The docstring's counts were wrong, and one of them was never a fact.** It said "71
   work on volumes and 33 do not", which is 104; on the same albumentations 2.0.8 the old
   list returned 130. The corrected list is **118 transforms**, and a test parses that
   number out of the prose and compares it to what the function returns - `albumentations`
   is pinned only as `>=1.4`, so an upgrade now names the new value instead of letting the
   documentation drift again, which is how the previous numbers survived two corrections
   elsewhere.

   The volume count is **not pinned, because it is not portable**, and that was found by
   pinning it: the first version of this test asserted all three numbers and one job of
   twelve answered "the docstring says 118/87/31; measured 118/88/30". The rank-3 list is
   produced by running each transform against a probe, and **eleven of the thirty-one it
   rejects fail for the probe's reasons rather than albumentations'** - nine require three
   channels where the probe has one, and `Crop`, `FrequencyMasking`, `TimeMasking` and
   `Superpixels` fail on its 8x8 size - so which of them tips over is environmental.

   Which job it was is worth stating precisely, because the obvious reading is wrong: 87 on
   all four Linux builds, on both macOS builds **and on Windows with Python 3.10**, and 88 on
   Windows with Python 3.13 alone. Neither axis explains it, and no mechanism is claimed.
   So the prose states the portable number, says the other one is not, and explains why;
   what is asserted about rank 3 instead is that it is a strict, non-empty subset of rank 2.

   Accounting, so the change is checkable: 130 -> 118 removes the 8 composition classes, the
   2 dataclasses, and `ImageOnlyTransform` and `Transform3D`, which the name filter had
   admitted. On volumes the old 97 becomes 87 here, the difference being the 10
   non-transforms, which had been passing the volume probe spuriously.

### Noted, not fixed

 - **The rank-3 probe is one channel and 8x8**, which is why eleven transforms are reported
   as unable to take a volume when what they cannot take is the probe. Enlarging it is not
   obviously right: a three-channel probe would list `ChromaticAberration` and its eight
   relatives as available to someone whose CT has one channel, and the listing cannot know
   the channel count - only the pipeline-build check can, and it already probes at the
   model's own input size. The size half (`Crop`, `FrequencyMasking`, `TimeMasking`,
   `Superpixels`) has no such objection and is worth doing on its own.

## [0.7.0a1] - 2026-10-07

One name per concept, across both packages. The R surface and this engine had grown two
words for several of the same things - `available_augmentations` against
`available_transforms`, `save_weights` against `export_weights`, `mask_report` against
`inspect_masks` - and a reader moving between the halves paid for every one of them. The
rule settled on is that **the engine's name wins**, because it is the layer that cannot be
renamed later without touching the configuration format and the schema; the exception is
where the R package already had a family of names, and `available_*` was one.

### Changed

 - **`known_weights()` is now `available_weights()`.** The R package already called it
   that, `available_transforms()` is the same verb, and "known" said nothing the other
   two did not say better. A breaking rename rather than an alias: two names for one
   thing is the thing this release removes.

### Added

 - **`available_transforms()` is exported from the package root.** It was reachable only
   as `pyplatypus.spec.components.available_transforms`, while the refusal a user meets
   when a 3D augmentation is unsupported tells them to "see `available_transforms(rank=3)`"
   and does not say where it lives. `from pyplatypus import available_transforms` raised
   `ImportError`.

 - **`tests/test_public_api.py`.** Four checks on the front door, none of which existed:
   every name in `__all__` resolves, `__all__` is sorted, both listings are reachable, and
   - the general form of the bug above - **every function a user-facing message tells
   somebody to call is importable from `pyplatypus`**. Each was proven able to fail by
   mutation; removing the `available_transforms` import fails three of them, naming
   `augmentation.py` as the file whose message would have lied.

### Not changed, and deliberately

 - `onehot_to_colours()` keeps its name. Pairing it with the R package's `mask_colours()`
   was proposed and measuring killed it: this one takes a one-hot or probability array and
   argmaxes it, R's takes class indices. Two names because they are two functions.
 - `Engine` / `DetectionEngine` / `build_engine` keep theirs. The R package has no name
   for the engine at all, which is a difference of idiom - an object with methods against
   verbs on a fitted model - and not a divergence to resolve.

## [0.6.0a4] - 2026-10-07

### Added

 - **`swa`: average the weights over the last part of the run.** Closes
   maju116/pyplatypus#25.

   ```yaml
   callbacks:
     - name: swa
       start: 0.75          # the fraction of the run after which averaging begins
       learning_rate: 1e-3  # held while it averages
   ```

   **What it is worth, and it is not what a table of scores shows.** Three seeds, 60
   epochs, lesions with ambiguous edges:

   ```
                Dice             volume bias        |volume error|
   plain        0.9639 ±0.0100   +1.71% ±12.18%     12.60% ±4.19%
   swa 0.75     0.9667 ±0.0087   +2.08% ±2.15%      12.38% ±4.31%
   swa 0.5      0.9640 ±0.0085   +2.78% ±3.44%      13.75% ±4.21%
   ```

   Dice does not move and neither does the per-case error - both inside the seed noise.
   What moves is **reproducibility**: the volume bias of the plain runs was +4.5%, +12.2%
   and -11.6% across seeds, and with averaging -0.1%, +4.2% and +2.1%. A spread **5.7
   times tighter**, and paired per seed the distance from unbiased improved on all three,
   by 7.34% at 4.9 standard errors. Which is what averaging is for: a point in the middle
   of a flat region is the same point whichever corner the last epoch wandered into.

   **Batch-normalisation statistics are recomputed** by a pass over the training data, and
   that is the step this feature mostly consists of. An averaged weight tensor *inherits*
   the statistics of whichever epoch was last rather than averaging them, so without the
   pass the model is evaluated under the wrong normalisation and scores far worse than it
   should with nothing to say why. `TrainingState` gained `train_loader` for it - a
   callback that cannot reach the data cannot do SWA honestly.

   **`swa` with `cosine_annealing` is refused.** Averaging needs the weights to still be
   moving; a cosine has decayed to its floor by the time averaging begins, so the average
   is just the last epoch and the callback did nothing - configured, silent, and
   indistinguishable from not having asked. `swa.learning_rate` is the other half of the
   recipe. `reduce_lr_on_plateau` is not refused: it lowers the rate on evidence rather
   than on schedule and may never fire.

   Nothing is replaced when a run ends before averaging began - early stopping can do that,
   and substituting one epoch's weights for "the average" would be a lie about what
   happened.

## [0.6.0a3] - 2026-10-07

### Changed

 - **`EngineBase`: the half of the two engines that is the same work.** Closes
   maju116/pyplatypus#100, which asked for the split into a segmentation and a detection
   engine plus a gate for future ones plus shared methods in a base. The first two shipped
   earlier - two classes, and `task` as a required discriminated union, which turned out to
   belong in the specification rather than in a class hierarchy. This is the third.

   **The width was measured rather than chosen.** Across the two engines twelve method
   names are shared and only two were the same work: `__init__` at 0.70 similarity and
   `_discover_test` at 0.65, the latter differing in nothing but its docstring and the name
   of one hook. `predict` is 0.05 similar and equal in length by coincidence. So the base
   holds split discovery, the test-split fallback, and the two refusals that go with them -
   and nothing else, because pulling up `predict` or `dataset` would produce a method that
   is mostly a branch on its caller.

   The argument was never tidiness. These two had already diverged twice: detection recorded
   which splits carry annotations while segmentation threw the same fact away, and the guard
   order that let "no masks" stand in for "no split" was wrong in both. The second time, the
   same conditional was written twice on one night.

   ```
   Engine            22 -> 20 methods      __init__ similarity 0.70 -> 0.61
   DetectionEngine   28 -> 27 methods      the discovery block: ~25 lines -> 2, in both
   ```

   A subclass provides `_has_any_labels()` - masks and annotations are found in different
   ways - and `labels_are`, the one word the shared refusal has to name. `tests/test_engine_base.py`
   asks the same question of each and compares the answers, which is the test that was
   missing; reverting the base's own logic breaks three of them across two files.

## [0.6.0a2] - 2026-10-07

### Added

 - **`validation: false`: train on everything and measure nothing, said out loud.** The
   third and last way of answering the question `validation_path` and `split` answer.

   ```yaml
   data:
     train_path: images/
     validation: false
   ```

   A final fit on every case you have, once the hyperparameters are settled, is a legitimate
   thing to want - and until now it meant inventing a split and ignoring the number it
   produced. Closes maju116/pyplatypus#44, whose third case was exactly this.

   **Silence is still an error.** A specification with neither a path nor a split is refused
   as before, and the message now names this option: "I have no validation set" and "I
   forgot" look identical, and only one of them is a decision. Giving `validation: false`
   *and* a path or a split is refused too - they say where the validation set comes from and
   this says there is not one.

   The validation split is then **absent rather than empty**. An empty one makes every count
   read zero and every average read nan, which is a number for something nobody asked for;
   a missing split makes `evaluate` say so.

 - **A callback cannot wait for a number that will never arrive.** Early stopping on
   `val_loss` in a run with no validation trains to the last epoch while waiting, and
   checkpointing writes nothing - both silently. Refused where the model and the data are
   both visible, since neither alone can answer it.

### Fixed

 - **"no masks" and "no split" are different things, and the messages had them confused.**
   `evaluate()` defaults to `"validation"`, so a run specified with `validation: false` met
   *"the 'validation' split has no masks, so there is nothing to score against"* - true of
   nothing, and it sends somebody looking for files they never wrote. Both engines now ask
   whether the split exists before asking whether it is labelled.

## [0.6.0a1] - 2026-10-06

A minor bump rather than another alpha letter: the data path grew a third item and the
loss interface grew a flag, both of which a custom loss or loader would notice.

### Added

 - **`boundary`: a loss that knows how far a wrong voxel is from the truth.** Dice and IoU
   count a voxel the same wherever it sits, which is why a model can reach 0.88 on Dice
   while its volumes run a fifth too large - the overshoot is all at the boundary, and for
   a small lesion the boundary is most of the object.

   ```yaml
   loss:
     name: boundary
     region: {name: focal, gamma: 2.0}    # or dice, iou, tversky, ...
     alpha: 0.9
   ```

   `mean(phi * p)` added to a region loss, where `phi` is the signed distance to the
   truth's boundary - negative inside, positive outside. Closes maju116/platypus#37, which
   asked for focal combined with a distance metric; `region` is how any of the nine is
   combined with it rather than one hard-coded pair.

   **What it buys and costs, measured.** Three seeds, 60 epochs, lesions with ambiguous
   edges, against Dice alone:

   ```
                    Dice             volume bias       volume |error|
   dice alone       0.9525 ±0.0027   -5.18% ±3.11%     16.90% ±0.90%
   boundary a=0.9   0.9411 ±0.0093   -0.06% ±4.15%     22.17% ±2.88%
   ```

   It trades accuracy per case for being unbiased over a series. "Is there a lesion" wants
   the overlap; "has it grown since March" wants a volume that is not systematically wrong,
   and no single number answers both. The bias improvement is suggestive on three seeds
   (the baseline's own bias wanders ±3.11%); the costs are established.

   **`alpha` defaults to 0.9, not 0.5, because 0.5 was measured and is unusable**: at that
   weight the surface term has half the objective while the prediction is still random, and
   three seeds gave a volume error spread of ±28.28% with every number worse than the
   baseline. Kervadec schedules it downwards from near 1; 0.9 is the closest a single
   number comes.

### Changed

 - **The data path can carry a signed distance map, and the loss decides.** A loss sets
   `needs_distance`, `loss_needs_distance(spec)` answers it from the specification, the
   loader computes the transform **in its workers** and the batch arrives with three items
   instead of two.

   In the loss it would have dominated training: one transform costs 5.3 ms at 256x256,
   23.4 ms at 64x64x32 and 709.8 ms at 128^3, against an epoch of about half a second for
   a small 3D model. The paths that score or predict rather than optimise never call the
   loss, so they ask for two items and pay nothing.

   It is recomputed every epoch rather than cached: augmentation moves the mask, and a map
   cached against a sample index would describe a shape that is no longer there - silently,
   since nothing ever looks at it.

 - **`scipy` is a direct dependency.** It was always installed, because albumentations
   requires it, but a package whose code imports scipy should say so rather than rely on
   somebody else's dependency list not changing.

## [0.5.0a6] - 2026-10-06

### Added

 - **`crop_boxes()` and `DetectionEngine.crops()`: every detection cut out of the image it
   was found in.** The pipeline is a detector followed by a classifier it does not
   contain - find objects with three classes, crop them, hand the crops to something that
   knows eighty. `predict` returns boxes in each image's own pixels precisely so they can
   be used on the photograph they came from; this is what uses them.

   ```python
   for found in engine.crops("bccd", "test", size=(224, 224)):
       batch = np.stack(found["crops"])      # one shape, ready for a classifier
   ```

   Five decisions, each of which taken the other way quietly changes what a downstream
   model sees:

   - **Cut from the source image, not the network input.** `DetectionDataset.source_image`
     re-reads at native size through the same reader and the same options as `read()`, so
     a crop and a prediction cannot disagree about channels or the DICOM window. Cropping
     from the letterboxed frame would cut a downscaled object and then scale it back up.
   - **Rounded outward**, so the crop contains the whole box. Rounding to nearest loses up
     to a pixel per side, a twelfth of a 26-pixel platelet.
   - **`context` expands by a fraction of the box's own size**, default 0, so one value
     suits a platelet and a white cell and nothing is expanded silently.
   - **`size` letterboxes rather than stretches.** Resizing a tall box to a square changes
     the aspect of every non-square object; `fit="stretch"` exists and must be asked for.
   - **Always a list**, even when `size` makes the shapes equal. A return type that depends
     on an argument makes every caller branch, and `np.stack(crops)` is the batch.

   `crops()` filters at `operating_point` rather than `score_threshold`, which is near zero
   so that average precision can integrate the whole ranking - cropping that would hand a
   classifier the tail AP exists to measure over.

   It also clips and `drop_degenerate`s first, reporting `dropped`. That was found by the
   tests rather than reasoned about: a model may predict a box off the frame and `predict`
   reports what it said, so at a low threshold some arrive with no extent at all once the
   letterbox is undone. The same two steps `DetectionDataset.read` already applies to
   truths.

   Closes maju116/platypus#95.

## [0.5.0a5] - 2026-10-06

### Added

 - **`box_loss`: score the boxes as `1 - GIoU` instead of as YOLOv3's offsets.** Default
   `offsets`, which is what the published `bccd-yolo3` weights were trained with, so nothing
   changes unless asked.

   **It is not the better objective, and that is measured.** BCCD's own test split, three
   seeds each, 150 epochs:

   ```
               mAP@0.5          mAP@COCO         matched IoU
   giou        0.8666 ±0.0147   0.5078 ±0.0083   0.7929 ±0.0069
   offsets     0.8583 ±0.0167   0.5172 ±0.0195   0.8038 ±0.0001
   ```

   Average precision cannot separate them in either direction - both gaps are inside the
   noise. Matched overlap can, and `giou` is worse by 0.0109, which is 2.7 standard errors
   of the difference; it is also sixty times less repeatable on exactly that quantity
   (`offsets` gave 0.803853 / 0.803656 / 0.803861 across its seeds).

   The reason is in the same runs' numbers: GIoU exists because IoU is a flat zero for boxes
   that do not touch, and anchors fitted by k-means already cover BCCD's truths at a mean IoU
   of **0.877**, so predictions never start disjoint and the advantage never arrives. Where
   fitted anchors cover poorly it should be a different story, which is untested and therefore
   not claimed.

   **What it is for is reading the loss.** The offsets term cannot reach zero - cross-entropy
   against a soft target bottoms out at that target's entropy - so all three runs converge to
   5.90-5.94 against a measured floor of 5.891, and a converged run prints what a stalled one
   prints. Under `giou` a prediction constructed to be exactly right scores `0.000000`, and on
   real data the term reads 0.05 on training against 0.47 on validation: a localisation gap
   that is visible because zero means zero. On BCCD that legibility costs about 0.011 of
   matched IoU.

   `_giou` agrees with `torchvision.ops.generalized_box_iou` exactly (0.0e+00 over 42 random
   pairs), with the cases baked into the tests so the suite needs no torchvision.

   Closes maju116/platypus#27.

## [0.5.0a4] - 2026-10-06

### Fixed

 - **Anchors may be a grouped array, and the flattened form is refused by name.**
   `_flatten_anchors` opened with `if not anchors:`, which an ndarray answers with
   `ValueError: the truth value of an array is ambiguous` - so the guard whose only job is
   to explain empty anchors reported numpy's complaint about emptiness instead, and a
   grouped `(3, n, 2)` array, the natural type, could not be passed at all.

   Two things made it reachable rather than theoretical: `encode` and `decode` are public,
   and the package offers the wrong-typed value under an inviting name - `AnchorFit.flat`
   is an `(N, 2)` array and is what one reaches for after fitting anchors.

   ```
   grouped tuple / list / (3, n, 2) ndarray   accepted, identical output
   empty list / empty ndarray                 "anchors is empty; YOLOv3 needs one group ..."
   AnchorFit.flat / a list of N pairs         refused, naming `.anchors` not `.flat`
   ```

   A flat `(N, 2)` array is refused rather than divided by three. It has thrown away how
   the anchors split between the output grids, which is half of what the function returns,
   and three equal groups is right for YOLOv3 and wrong for a two-head model; an anchor
   assigned to the wrong stride shows up in no metric.

   The R surface already refused a flat matrix by name, so this brings the engine up to the
   standard the visible half already set. Found while reading the issue tracker, where a
   2022 report of the same mistake in R - `the condition has length > 1` - was being closed.

## [0.5.0a3] - 2026-10-06

### Added

 - **`split`: divide the training folder instead of naming three paths.** A researcher with
   one directory had to run a splitting tool, point three paths at its output, and remember
   to switch mode - and when they did not, the refusal said only `validation_path: Field
   required`, which is true and names nothing they could do about it.

   ```yaml
   data:
     train_path: images/
     split:
       fractions: [0.8, 0.2]          # two, or three to cut a test set as well
       group_by: "^(patient\\d+)_"     # required key, may be null
   ```

   Exactly one of `split` and `validation_path`, for the reason `colormap` and `labels` are
   exactly one: two sources for one thing are two places to change it.

   **`group_by` is required and may be `null`.** Everything sharing a group lands in a
   single split - usually a patient. It is required rather than optional because the
   mistake it prevents leaves no trace: slices of one patient in training and validation at
   once make validation measure memory rather than generalisation, and the score comes out
   several points too high with nothing in the output to say so. An omitted key would be
   that choice made silently; `null` is the same choice made on purpose. `splits.py` has
   refused a pattern that *matches* nothing since it was written, for the same reason; this
   extends it to never having asked.

   **Nothing is written.** `split_samples` partitions what was already discovered.
   `split_dataset()` is still the tool when the CSV files are the point - to keep, to hand
   to a colleague, to cite - but being read should not leave files behind.

   **A test set cut this way is scoreable**, because it came from annotated data. Unlike a
   separate `test_path`, which may be images alone and is recorded as unlabelled, the third
   fraction can be evaluated and not only predicted on.

   The two-or-three rule is not restated in the spec: it calls `_fractions` from the
   splitting module, which already had it.

## [0.5.0a2] - 2026-10-06

### Fixed

 - **Segmentation now records which splits carry masks, as detection always has.** It
   discovered the fact and threw it away: a test split was always found with
   `only_images=True`, so two things were wrong at once and in opposite directions.

   A test folder of images alone built a dataset that would try to read masks and fail on
   the first item with `MaskError: no masks to unite` - true, two layers below the question
   that was asked, and about the wrong thing. `evaluate("test")` now refuses by name: *"the
   'test' split has no masks, so there is nothing to score against. predict('test') works on
   it."*

   And a test folder that **did** carry masks was never recognised as scoreable, because
   discovery never asked for them. `evaluate("test")` on a complete test set now simply
   works, where before it failed exactly as it did on an empty one.

   `dataset(..., only_images=...)` defaults to what the split actually has rather than to
   `False`. Passing it explicitly still overrides.

   **"No masks" and "some masks" stay different.** Falling back whenever labelled discovery
   failed would turn a split with two missing files into an unlabelled one, discarding the
   seventy that are there and reporting nothing - so the fallback happens only when the
   split carries no masks at all, and an incomplete one raises. Taken from the detection
   side, which met this first.

## [0.5.0a1] - 2026-10-05

**Breaking.** Three fields in a model block were assertions wearing the clothes of settings -
each one restating a fact that lives somewhere else, able only to agree or to be wrong.

### Removed

 - **`n_class` is no longer a model field.** The number of classes comes from the data, from
   `colormap` or `labels`, and always did. The proof that the field was wrong is what left with
   it: a spec-level validator existed for the sole purpose of catching a model disagreeing with
   its own data. There is nothing left to disagree.

   The count still reaches the weights sidecar - contributed by the engine, from the half of
   the specification that knows it - so weights trained for a different number of classes are
   refused exactly as before.

   `build_model(spec, n_class=...)` takes it as an argument now: a network's head is that wide,
   so it has to be told, and saying so in the signature is honest.

### Changed

 - **`channels` is derived from `channels_from` when the model does not state it.** Unlike
   `n_class` the field stays, because without `channels_from` it is a real choice - the same
   files can be read as one channel or as three. Stated, it is still checked and still refused
   when it disagrees; silent, it is filled in.

 - **A weights file's own geometry is adopted.** `architecture`, `blocks` and `filters` - and
   `anchors_per_grid` for a detector - are taken from the sidecar where the specification
   stayed silent. Using somebody else's published model meant knowing its internals
   (`dsbowl-unet` is four blocks of sixteen filters) and being refused for guessing. Detection
   has adopted its anchors this way since 0.3.0a12; this is the same move applied to the rest
   of the fingerprint.

   `model_fields_set` is what makes it safe: a value you wrote and a default that happened to
   be there are the same number and not the same claim. Anything stated is left alone and still
   has to agree, so this loosens what may be omitted and nothing about what is checked.

   **`input_shape` and `channels` are deliberately not adopted.** The rank comes from
   `input_shape` and is needed while the specification is validated - before any sidecar can be
   reached without a download - and the data pipeline reads both before a network exists. A
   specification that cannot be checked offline is the air-gapped hospital problem this project
   has carried since RECON.md. Pinned by a test, so it reads as a decision rather than a gap.

### Migrating

Delete `n_class` from every model block; nothing replaces it. If the count was right it was
redundant, and if it was wrong the run was already being refused.

## [0.4.0a1] - 2026-10-05

**Breaking.** A minor bump rather than another alpha letter, because the configuration format
changed and a number that hides that is a number doing nothing.

### Changed

 - **`task` is required.** It was defaulted to segmentation so that files written before
   detection existed kept working. The default is gone: a configuration that does not say what
   it is asking for means whatever version reads it, and more tasks are coming.

   The default also sent the bill to the wrong person. A *detection* configuration missing its
   `task` was validated as segmentation and reported `classes` and `anchors_per_grid` as **extra
   fields**, never naming the tag - exactly the confusion a discriminator exists to prevent.

 - **Both task names are spelled out**: `segmentation` is now `semantic_segmentation` and
   `detection` is now `object_detection`.

   `segmentation` stops naming one thing the moment instance segmentation exists, and a value
   that has to be reinterpreted later is worse than a longer one now. These are also the two
   words the 2022 package used for its own top-level configuration keys, so this is a return
   rather than an invention.

   **Both old values are recognised by name**, because they are the only strings an existing
   file or an older script can contain:

       task: segmentation  ->  "`task: segmentation` was renamed to `semantic_segmentation`."
       task: detection     ->  "`task: detection` was renamed to `object_detection`."
       (missing)           ->  "a configuration has to say what it is asking for: add `task`,
                                one of 'semantic_segmentation', 'object_detection'."

   Left to pydantic, each of these would have been an "input should be ..." list that the
   reader has to decode. A rename the reader must deduce is a rename done to them.

### Migrating

Add one line to every configuration, and change it if it was already there:

```yaml
task: semantic_segmentation     # or object_detection
```

Nothing else moves. The R package builds `task` from its own constructors and sends it, so R
code needs no change beyond pinning an engine of this version or later.

## [0.3.0a15] - 2026-10-05

### Fixed

 - **0.3.0a14 reports itself as 0.3.0a13, and this release is that release with its version
   corrected.** The version lives in two places - `pyproject.toml` and
   `pyplatypus.__version__` - and 0.3.0a14 bumped only the first. Nothing failed: the wheel
   built, the metadata was right, 942 tests passed and the release went out. The only visible
   symptom was the R package refusing to believe the engine was the version it had pinned,
   which is that check doing exactly what it is for.

   It matters beyond cosmetics because `__version__` is what gets written down. `write_record()`
   stamps it into every run record and `export_weights()` into every weights sidecar as
   `trained_with`, so a wrong value is wrong provenance in files that outlive the release.
   **Anything produced by 0.3.0a14 is labelled 0.3.0a13.**

   `tests/test_version.py` now pins the two literals to each other, and to the installed
   distribution's metadata when there is one. Every release from 0.3.0a11 to 0.3.0a13 happened
   to keep them in step; nothing required it.

   Use 0.3.0a15. 0.3.0a14 works - the only thing wrong with it is the number it gives for
   itself - but a package that misreports its own version has no business being pinned.

## [0.3.0a14] - 2026-10-05

### Added

 - **`evaluate_images()`: one row per image for detection.** The counterpart of the
   segmentation engine's per-case scores, and the question a table of averages cannot reach -
   not how well on average, but *which* images it fails on. A mean over a split says 0.86; it
   does not say the misses are concentrated in four frames where the stain is dark, and that
   difference is usually about the data rather than the model.

   Columns are `key`, `n_truth`, `n_predicted`, `matched`, `missed`, `spurious` and
   `mean_matched_iou`. Sort by `missed` for the frames it cannot see, by `mean_matched_iou` for
   the ones where it sees everything and places it badly - different problems, usually the data
   and the anchors respectively.

   **There is deliberately no average precision per image.** AP is the area under a
   precision-recall curve and therefore a property of a ranking over a whole dataset; computed
   on one image with three boxes it swings on a single box's rank and means nothing - the same
   degeneracy as a per-case Dice on an empty truth mask. What is meaningful for one picture is
   counting and overlap, so that is what the row carries.

   **Counts are read at the specification's `operating_point`**, not over the whole ranking.
   "How many were missed" is undefined at a threshold of zero, where every box the model dimly
   considered counts as a prediction and `spurious` would count the tail that average precision
   exists to integrate over.

   `mean_matched_iou` is `None` rather than `0.0` when nothing matched: an image where the model
   found nothing and one where it found badly-placed boxes are different failures, and a zero
   merges them.

   Matching reuses `match_detections`, the same function the dataset-wide report uses, so the
   rows are a decomposition of the table rather than a second implementation - pinned by a test
   that asserts they sum to it. The example prints the five worst frames and how many were
   exactly right.

## [0.3.0a13] - 2026-10-04

### Fixed

 - **The colormap was described and not recorded.** `pyplatypus.weights` has said since it was
   written that it cannot catch weights trained on a different colormap with the same class
   count - "those load cleanly and predict nonsense" - and it could not, because the colormap
   went into no sidecar. It was not inherent: the detection side already compared its class
   *names*, which is the same check. The colormap, or the labels for a label-map dataset, now
   go beside the weights and are compared on load.

   Weights published before this keep working, by construction rather than by a version check:
   the comparison skips a field the sidecar does not carry, and `dsbowl-unet` carries none.
   Pinned by a test rather than reasoned about.

### Added

 - **A run leaves a record.** The weights are not the whole of a trained model: a segmentation
   model is meaningless without the colormap, and a detector without the anchors its boxes are
   relative to - which, when they were fitted rather than named, existed only in memory and in
   a weights sidecar somebody had to remember to export.

   `fit()` now writes `<output_dir>/<model>/run.json`, **only when `output_dir` was given**.
   `model_fields_set` answers that honestly: a field with a default cannot otherwise be told
   apart from one the user set to that default, and writing files into somebody's working
   directory because a default exists is not a thing to do by surprise.

   The record is the **specification** plus what the run derived from it, rather than a
   summary: a summary is for reading and a specification is for running again. For detection
   that means the anchors, whether they were fitted, their coverage and the target survey; for
   segmentation `derived` is empty, which is the honest answer rather than an omission - the
   specification already carries the colormap, the input shape and the window.

   The test that matters puts the specification and `derived.anchors` back together and checks
   the engine used **those** anchors rather than fitting new ones.
 - `shape_table()` and `DetectionEngine.box_shapes()`: every annotated box as a width and a
   height with the class it belongs to, beside the anchors in use, in one call and in one set
   of coordinates. What a picture of the anchor fit needs - a cloud of shapes coloured by class
   says whether a class has anchors near it at all, which no summary statistic does.

   `box_shapes()` and `shape_table()` share one implementation of the letterbox arithmetic,
   deliberately: two copies is the duplication that reads as harmless and ends with a plot
   showing boxes in different places from where the anchors were fitted to them - which looks
   like a bad fit rather than like a bug. A test holds them to the same numbers.

   `box_shapes()` keeps its old contract and does **not** require labels. The first version of
   the shared helper read them unconditionally and broke every caller standing in for an
   annotation with boxes and a frame.

### Fixed

 - `LossParts.as_dict` turned tensors that still tracked gradients into scalars, which torch
   2.14 warns about. **Invisible in this package's own environment**: the venv here pins torch
   2.7.1 for a Pascal card and does not warn, while `py_require()` in the R package resolves a
   current torch and does - so the R test suite is what saw it. The same mistake had already
   been made and fixed in `training/trainer.py`, whose comment says exactly this.

   Not covered by a test, deliberately: the value is identical either way and the warning does
   not fire on the torch installed here, so a test would pass with the fix removed.

## [0.3.0a12] - 2026-10-04

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

 - **`bccd-yolo3` in the weights registry**: YOLOv3 at 416x416 for RBC, WBC and Platelets,
   trained on BCCD (MIT) using the dataset's own split, pinned to commit `24fbff08`.
   `mAP@0.5` 0.857 on the 72 held-out images - the **median of five seeds**, not the best,
   because the spread is 0.0159 and reporting one run would be reporting whichever seed was
   chosen. The card states what decides whether the weights answer someone's question:
   **platelets are the unreliable class**, precision 0.54 at confidence 0.5 and four times
   the seed-to-seed variance of the other two.

   Verified by fetching it by name from the Hub on a cold cache and scoring it: 0.8571 /
   0.5003 / 0.8058, equal to the card and the sidecar to four places
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
