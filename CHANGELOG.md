# Change Log
All notable changes to this project will be documented in this file.

## [0.8.0a37] - 2026-10-24

A fourth callback hook, and the averager that needed it. `maju116/rplatypus#203`.

### Added

 - **`Callback.on_batch_end`**, fired after every optimiser step on the training pass,
   with `state.step` counting steps since training began. The protocol had three moments
   and none of them was a step.

   **It cannot stop the run**, and returns nothing to say so. A run that stopped mid epoch
   would leave a record with training numbers and no validation ones, and every reader of
   the history - `best()`, the callbacks that watch it, R's `training_history()` - would
   have to learn that a row may be half a row. `stop_after`'s budget is therefore still
   the budget plus one epoch, which its own description says.

 - **`ema`**, an exponential moving average of the weights updated every step. `swa`'s
   sibling and the reason the hook exists: a decay of 0.999 is 0.999 *per step*, which
   over 60 epochs of 200 batches is 12,000 updates and a half-life of about 700 steps.
   Announced per epoch the same number would do almost nothing, so offering it that way
   would have published a name whose documented values mean something else.

   Buffers are averaged alongside the weights, which is the difference from `swa`: a
   uniform average of weights has batch-normalisation statistics belonging to no epoch, so
   `swa` recomputes them with a pass over the training data; an exponential average that
   also averages the buffers carries its own and needs no second pass.

 - **One averager per run.** `ema` and `swa` together are refused: both replace the model
   when training ends, so the second would average what the first had already written, in
   whichever order the list happened to have them.

### Measured

 - Five mutations, each caught: firing the hook on validation batches fails 1 test, a step
   counter that never advances fails 1, an average that is counted but not folded fails 2,
   a run with no steps that still gets the average fails 1, and two averagers allowed
   fails 1.

 - **One of those was not caught the first time, and the test was the problem again.**
   `AveragedModel` is built from the model at `on_train_begin`, so with the weights
   untouched its snapshot and the model are the same numbers and writing one over the
   other is invisible. The test now moves the weights before asking whether they were left
   alone. The same shape as `0.8.0a35`'s range assertion: a check that cannot tell the
   failure from the success.

## [0.8.0a36] - 2026-10-24

The number this package tells everybody to read becomes one a callback can watch.
`maju116/rplatypus#205`.

### Added

 - **`val_<metric>_worst` in every epoch**, for the metrics whose counts pool into one
   score: the lowest any single case scored. `summarise_cases` has said since it was
   written that *"the minimum is the one nobody publishes and everybody should - a model
   averaging 0.86 that scores 0.11 on some patient has a failure mode, and the mean is
   where it hides"*, and then every callback in this package watched a mean. A declaration
   beside a mechanism that did not describe it.

   `monitor: val_dice_worst` works now, and so does a `when` condition on it.

   **Validation only.** The training loader is shuffled, so a case has no identity in that
   pass and there is nothing honest to pool by.

### Changed

 - `score_cases` and the epoch share one definition of a case's score, `_PerCase`. The
   pooling is a ratio of sums - a case's tiles, or a patient's slices, are added up and the
   formula applied once - and that arithmetic now exists in one place rather than in the
   two that report it.

### Measured

 - **The worst case costs 5.6% of an epoch**, on 40 cases at 128x128 on this machine's
   processor: 620.9 ms of training plus 366.3 ms of validation becomes 1042.6 ms. It is
   computed in the pass that already happened - the counts are three reductions on a batch
   that has been forwarded anyway - rather than in a second pass, which would have doubled
   every epoch's validation to report a number the first one had the parts for.

   The share *of the validation pass alone* came out 23.3% and then 15.1% on two runs of
   the same measurement, so the epoch figure is the one worth quoting and the other is
   noise.

 - Five mutations, each caught: reporting the best instead of the worst fails 4 tests,
   replacing a case's accumulation with its last piece fails 1, not noticing more examples
   than names fails 1, not reporting it at all fails 5, and a spec that does not admit the
   key fails 1.

## [0.8.0a35] - 2026-10-24

The quiet failure of segmentation becomes a number, and then a refusal.
`maju116/rplatypus#206`.

### Added

 - **Every epoch reports `foreground`**, for both passes: the share of the prediction that
   is not background. Beside the loss rather than among the metrics, and the distinction is
   the point - nothing is compared against the truth, so this is not a measure of quality.
   It answers *is this still a segmentation* rather than *is it a good one*.

   Reported unconditionally rather than offered as a metric to opt into, because **a
   collapse you have to opt into noticing is one you will not notice.** A model predicting
   background everywhere is not visible in the numbers it already produced: with
   `include_background` the Dice is a mean over class channels and the background channel
   is nearly perfect, so a collapsed model reports around 0.5 while its loss falls.

 - **`terminate_on_collapse`**, `terminate_on_nan`'s quiet sibling: it ends the run when
   the foreground share stays at or below `floor` for `patience` epochs. A floor rather
   than a test for zero, because 0.01% foreground on a target that is 14% foreground has
   collapsed in every sense that matters; and a patience, because an early epoch
   legitimately predicts almost nothing. It can watch the training share instead, which
   answers a different question - a model collapsing on data it has seen has a different
   problem from one collapsing only on data it has not.

### Changed

 - `evaluate()`'s rows carry `foreground`. It is the one column there that is not a
   quality measure, and it is there for the reason `load_shape` is: a collapsed model
   reports a plausible Dice, and a table that cannot show the foreground beside it is a
   table somebody misreads.

### Measured while building it

 - **The test fixture collapses.** Three 32x32 images trained for a few epochs predict no
   foreground at all - share exactly 0.0 - which is a fair demonstration of what this
   release is for and makes that fixture useless for comparing the reported number against
   a computed one. That test evaluates an untrained model instead, and says why.
 - **One mutation was not caught the first time, and the test was the problem.** Deleting
   the accumulation entirely left every test passing, because the assertion was
   `0.0 <= share <= 1.0` and a counter that was never added to is zero - which is in range.
   The test compares values now, with the fixture's own share asserted non-zero first.

## [0.8.0a34] - 2026-10-24

Three answers to one question - *when may this callback act, and on what evidence* -
shipped together because they are one vocabulary rather than three features.
`maju116/rplatypus#199`, `#201`, `#207`.

### Added

 - **`early_stopping` takes a warm-up**: `start` as a fraction of the run or `start_epoch`
   as an epoch, exactly one of the two. `patience` alone could not express it, because it
   is in force for the whole run: a model that needs fifteen epochs before its validation
   curve means anything forced `patience > 15` permanently, paying the same slack at the
   end where the point was to cut quickly.

   **It delays the stopping and nothing else.** An epoch inside the warm-up can still be
   the best one and `restore_best` can still return to it - those are the best weights the
   run has if it ends for another reason, and discarding them to tidy the bookkeeping
   would lose a real answer.

 - **`swa` takes `start_epoch`** as the exclusive alternative to its `start` fraction. The
   same concept and therefore the same two words, by §3A. Its refusal reads
   `model_fields_set` rather than the value, because `start` carries a default and 0.75 is
   both the default and a perfectly good thing to ask for.

 - **`model_checkpoint` takes `when`**: thresholds that must all hold before anything is
   written, `{val_dice: 0.9, val_loss: 0.2}`. **The direction is derived, not configured** -
   anything ending in `loss` has to be below its threshold and everything else above it,
   which is the rule `monitor` already uses, so there is no `above`/`below` to get
   backwards. The keys are checked against what the model actually reports when the
   specification is built, the same validator that refuses an impossible `monitor`.

   `when` and `save_best_only` are **both gates**, not alternatives: with both, a file is
   written when the watched number improves *and* every threshold holds. And a run whose
   conditions never held **says so when it ends**, because silence there is
   indistinguishable from a callback nobody configured, and a run that kept no weights is
   worth knowing about before somebody goes looking for the file.

 - **`stop_after`**, a wall-clock budget in hours. An epoch cap is the wrong instrument:
   how long an epoch takes is a property of the data and the shape, and the cases where
   this matters are the ones where that was not what anybody expected. Checked at the end
   of an epoch, so the real limit is the budget plus one epoch - said in the field's
   description rather than discovered. The run's `stop_reason` names the clock, so a run
   that ran out of time does not read as a run that converged.

### Notes

 - A fraction and an epoch are **two fields** in both places, never one that accepts
   either: `1` and `1.0` are the same value once YAML is parsed, so "epoch one" and "the
   whole run" would be one configuration and the mistake would be silent - both are in
   range for both meanings.
 - Eight mutations, each caught: ignoring the warm-up fails 2 tests, letting it gate the
   best fails 3, reading a fraction as an epoch fails 1, turning the conditions into an OR
   fails 4, reading every condition as *above* fails 1, staying silent about a run that
   wrote nothing fails 1, a clock that never runs out fails 1, and an unchecked condition
   key fails 1.

## [0.8.0a33] - 2026-10-24

Two learning-rate schedules, and the discovery that the one we had could not be combined
with the thing its own docstring said it combined with.
`maju116/rplatypus#200`.

### Added

 - **`linear_annealing`** - the same shape as `cosine_annealing` with a straight line
   instead of a curve: `min_lr`, an optional `epochs`, each parameter group descending
   from its own initial rate.
 - **`step_decay`** - multiply by `factor` at fixed points, which is how published recipes
   are written: *divide by ten at epochs 30 and 60*. `every` for a period or `milestones`
   for the epochs a paper names, **exactly one of the two**, because `every: 30` and
   `milestones: [30]` are different schedules and one field holding either would be read
   as whichever the author happened to type. The rate is computed as
   `initial * factor ** drops` rather than by multiplying what is there, so replaying an
   epoch cannot double a drop.

### Fixed

 - **A schedule beside `reduce_lr_on_plateau` silently erased the plateau's cuts, and
   `cosine_annealing`'s docstring had promised the opposite since it shipped.** It said
   *"a run may want both: a cosine floor with a plateau rescue on top is a legitimate
   recipe"*. Measured instead of believed - the same six epochs, with a loss that never
   improves:

       epoch   both configured   the plateau alone
         1          0.097553           0.100000
         2          0.090451           0.010000
         3          0.079389           0.001000
         4          0.065451           0.000100
         5          0.050000           0.000010
         6          0.034549           0.000001

   The left column is the pure cosine from 0.1, digit for digit: a schedule recomputes the
   rate from the optimiser's initial value at the end of every epoch, so whatever the
   plateau cut is gone before the next one begins. The spec refuses the pair now, by name,
   and the docstring says what the code does.

 - **Two schedules in one model are refused** for the same reason, where before the last
   one in the list won silently.

### Changed

 - **`build_callbacks` declares an order instead of inheriting one.** `swa` pins a constant
   rate once averaging starts and a schedule recomputes it every epoch, so which one an
   epoch ended with depended on where they sat in the list and nothing said so. `swa` is
   moved last - a constant tail rate is the whole point of configuring it - and everything
   else keeps the order it was written in.
 - `cosine_annealing`, `linear_annealing` and `step_decay` share a base class, so the
   horizon, the per-group initial rates and the refusal to run without knowing the run's
   length exist once rather than three times.

## [0.8.0a32] - 2026-10-24

The honest comparison becomes a thing to ask for rather than a thing to write yourself.
`maju116/rplatypus#187`.

### Added

 - **`evaluate(at="source")` and `evaluate_cases(at="source")`**: every prediction goes
   back onto the grid of the file it came from and is scored against the truth **as
   stored**. `at="model"` is the default and is unchanged, because changing it would
   silently restate every number anybody has written down.

   What it is for: scored each through its own loader, a model at `input_shape: [512, 512]`
   is measured against a truth reduced to 512 while a model that tiles 2048 is measured
   against the truth as stored. Both are a real Dice of a real prediction against a real
   mask and they answer different questions. Measured on retinal vessels, that difference
   **reported a resized model at +0.0187 where the honest figure was +0.0033** - 5.7 times
   - with the bias reversed on clDice, because upsampling smooths a prediction: better
   against a skeleton, worse against a boundary.

   **The grid is the files' own and cannot be named by the caller.** A named grid resamples
   the *truth*, so the thing every model is judged against would itself become an
   interpolation chosen by whoever typed the number - which is the defect this argument
   exists to remove. Here the truth is never touched; only predictions move.

   **Streamed.** One case at a time through `predict_stream`, because the predictions are
   at full resolution and tiling exists for images too large to resize: 200 FIVES retinas
   at 2048 are 6.7 GB of probabilities, so this grid is exactly where holding the split
   would hurt most.

   **The spacing is each file's own**, which is what makes `surface_dice` mean millimetres
   here. On the model's grid every case shares `target_spacing` by construction; on this
   one they do not, so the metrics are built per case.

   Two prices, both named in the prose rather than discovered: interpolating probabilities
   softens a boundary, and where the forward crop cut anatomy away the inverse pads it with
   background - *not examined* rather than *nothing there*. For a comparison the second is
   correct rather than unfair, but a 0.000 arriving that way is a result and not a bug.

   The `load_shape` column reads `source` when scored there, because what it has always
   meant is the grid the row's numbers were computed on. The loss is not reported: a loss
   is defined by the objective a model trained with, on the grid it trained on.

### Changed

 - The comparability warning now names `evaluate(at='source')` instead of describing what a
   reader would have to build. It had pointed at `predict(space='source')` since
   `0.8.0a30`, which is the raw material rather than the answer - an instruction addressed
   to a human in the place where a mechanism now exists.
 - `Dataset.to_target()` is the three-way expansion `__getitem__` used to end with, and
   `Dataset.target_as_stored()` reads a sample's masks at the size its files hold. Reading
   `size=None` as *as stored* goes through the same reader as everything else: the three
   namings are the part that must not exist twice.

### Measured while building it

 - **`evaluate()` means over batches and `evaluate_cases()` pools by counts**, so a split's
   figure depends on `batch_size`. `at="source"` deliberately matches `evaluate()`'s
   pooling so that the argument changes the grid and nothing else; the two poolings differ
   by 4e-5 on six similar cases and by more on unequal ones. Filed as
   `maju116/rplatypus#196` rather than changed here, because changing it restates every
   number this project has published.
 - **The warning cannot be narrowed to physical extent, and does not need to be.**
   `target_spacing` lives on the data spec, so every model in one `evaluate()` shares it
   and the extent differs only by `load_shape`, which the warning already compares. The
   case that motivated the narrowing - three arms at one `load_shape` and three spacings -
   is three separate runs, which one engine never sees side by side. `at="source"` is what
   makes those comparable, because the files' grid is shared by construction.

## [0.8.0a31] - 2026-10-24

One defect, and it is the package disagreeing with its own naming convention.
`pyplatypus#256`.

### Fixed

 - **A checkpoint named `.safetensors` was written as a torch archive and then refused by
   the reader its own extension had selected.** `model_checkpoint` called `torch.save`
   whatever the path said; `load_into` chose `safetensors.load_file` when the suffix was
   `.safetensors`. So the round trip failed with `SafetensorError: Error while deserializing
   header: header too large` - an error about a header, raised on a file that has none, and
   naming neither the cause nor the field that caused it.

   **The configuration that breaks is the one in this package's own guide**, on both sides:
   `- {name: model_checkpoint, monitor: val_iou, path: runs/best.safetensors}` appears in
   `guide.qmd` and in the R package's `configuration` vignette. Every published set in the
   registry is `.safetensors`, so a person naming a checkpoint that way is following the
   convention this package teaches. Found while writing `#202`'s example, which had to work
   around it with a `.pt` and a comment.

   Fixed in both directions, because they are not alternatives:

     - `save_state()` writes the format the extension names, and is now the **one** writer -
       `export_weights()` and the checkpoint callback had been two, which is the defect
       stated from the other end.
     - `read_state()` reads what the file **is**, from the first bytes: a zip is a torch
       archive, an 8-byte header length followed by `{` is safetensors. So a file written by
       an older version of this package loads under whatever name it was given, and renaming
       a file never changes whether it can be read.
     - A file that is neither now says what it begins with, instead of letting a library
       report a header that was never there.

   Three mutations, each caught: a writer that ignores the extension fails 3 tests, a reader
   that trusts it fails 2, and a reader that returns `{}` rather than refusing fails 1.

   **And the same defect turned out to be inside torch, one layer down.** `torch.load` reads
   the *name* as well: on **torch 2.14.1** a path ending `.safetensors` is forwarded to
   `safetensors.torch.load_file`, while `torch.save` still writes a zip under that name - so
   the first fix failed every CI job on a suite that was green here, where torch is 2.7.1.
   `read_state()` hands `torch.load` an **open file** rather than a path, because a handle
   has no name to dispatch on. The fixtures were rebuilt the same way: an archive is written
   to a `.pt` and renamed, so the test does not depend on which side of torch's own
   disagreement wins, and it asserts the file really is an archive before reading it.

### Changed

 - `model_checkpoint`'s `path` description says the extension decides the format. The R
   package's prose for it is unchanged until its pin moves, which is when the field-prose
   ledger asks.

## [0.8.0a30] - 2026-10-23

Two things a table and a writer could not say, and the second one is `0.8.0a29`'s own half
finished. `pyplatypus#251`, `maju116/rplatypus#187`.

### Added

 - **`evaluate()` says which grid each row was scored on, and warns when the rows do not
   share one.** Each model is scored through its own loader, so a model at
   `input_shape: [512, 512]` is measured against a truth reduced to 512 while a model that
   tiles 2048 is measured against the truth as stored. Both numbers are a real Dice of a
   real prediction against a real mask, and they are not answers to the same question.

   Measured on retinal vessels: that difference reported a resized model at **+0.0187 where
   the honest figure was +0.0033** - the reported figure 5.7 times the real one - and the
   bias has the *opposite sign* on clDice, because upsampling a prediction smooths it and a
   smoothed vessel agrees better with a skeleton and worse with a boundary. So a comparison
   reporting one of the two metrics is wrong in a direction set by which one it picked.

   The `load_shape` column is permanent and the warning fires only when the grids differ,
   which is the split that keeps a warning worth reading. It sits beside the metrics for the
   same reason `loss_function` sits beside `loss` - it is what decides whether two rows may
   be compared, and a table without it cannot be read correctly a month later. A **string**
   rather than the tuple, because this table crosses into R as a data frame and a list
   column is not plain data; the test that says so caught it on the first run.

   Nothing here rescores. Putting two models on one grid is a thing to ask for rather than
   to do silently, and it is not offered yet - `rplatypus#187` carries that decision.

 - **`save_volumes` accepts a label map**, not only channels-last probabilities. The R
   package's `predict()` returns class indices where this one returns probabilities, so
   forwarding meant expanding a label map back into one-hot - a gigabyte of float32 for a
   512-cubed volume - in order to collapse it again. `#251`, and it is what lets
   `rplatypus::save_volumes()` forward and the copy in that package's shim be deleted.

   A label map **skips the overlap refusal, and that is not a hole**: the refusal protects
   the *collapse*, because an argmax over channels silently picks one of the classes a voxel
   is in. An integer per voxel cannot express an overlap in the first place.

   A bare rank-4 array is **refused rather than guessed at**: a stack of label maps
   `(n, d, h, w)` and one volume's channels `(d, h, w, c)` have the same rank and nothing in
   the array says which. `save_class_masks` names the same ambiguity for pictures and the R
   package resolves it the same way, so a caller meets one rule rather than a third.
   `save_class_volumes` still takes channels only, because a binary file per class is a
   question about channels and a label map has none.

## [0.8.0a29] - 2026-10-23

One writer, and it is the one this package's own refusal has been telling people to use
since `0.8.0a23`. `pyplatypus#249`.

### Added

 - **`save_volumes` and `save_class_volumes`**: a predicted volume written as NIfTI onto the
   geometry of the scan it came from. Integer labels, `int16`, with `reference` required
   rather than optional because there is no sensible affine to invent.

   **The affine is the point.** A mask array on its own cannot be laid over its scan: every
   viewer, every registration tool and every volume in millilitres reads position and
   spacing out of it, and a file carrying the wrong one is either silently misplaced or
   rejected. The reference is read through `_load_canonical`, the same function
   `read_volume` uses, because the reader reorients every volume to the closest canonical
   axes - so a prediction is in canonical space, and writing it with the file's **stored**
   affine would produce a mask reflected or transposed with respect to its scan, opening in
   any viewer without complaint. The test that earns its keep writes a scan stored the other
   way round and demands the mask read back equal; mutated to use the stored affine, it
   fails along with the affine test beside it.

   `save_volumes` **refuses an overlapping prediction** rather than taking the `argmax`,
   which is `onehot_to_colours`'s refusal one dimension along: a label map holds one integer
   per voxel, so a voxel in both the core and the whole tumour would be written as whichever
   channel read higher. Those are `regions`, and `save_class_volumes` is where they go - one
   binary volume per class, which is the writer BraTS actually needs, since `regions` was
   motivated by a dataset that stores its overlapping regions as volumes.

   It also refuses **case names that repeat**, which is not a theoretical tidiness: volume
   datasets keep one directory per case with the same `ct.nii.gz` inside each, so names
   taken from the sources collide, every mask is written to one path, and the count of
   written paths is still right. Losing five masks of six while reporting success is the
   worst outcome available here.

### Fixed

 - **`save_class_masks`'s refusal now names functions that exist.** It has said *"write each
   channel as NIfTI with the geometry of its source instead"* since `0.8.0a23`, and the only
   way to follow that instruction was to leave this package. The same shape as `#230` and
   `#231`, one step worse: those were promises about this package's own surface, and this
   was an instruction a reader could not carry out at all.

### Where this code came from

Not written from nothing. `rplatypus/inst/python/platypus_shim.py` has carried a correct
`write_volumes` since the volume work landed - Python, shipped inside the R package, doing
a thing this package's own reader is half of. A `pip install pyplatypus` user never had it.
The shim's copy is the one to delete when R's pin moves to this release; two
implementations of an affine decision is the shape that keeps costing this project.

## [0.8.0a28] - 2026-10-16

Two features and one lesson they share: a mechanism that existed twice, and a mechanism
that existed once and was reached through a list somebody had to remember.

`pyplatypus#236` - what a published file of instance weights may fill in for a reader, and
the sentence that explained why it could not. `pyplatypus#91` - one binary mask file per
class, which is the half of multilabel segmentation that 2026-10-12's decision deferred.

### Added

 - **An instance model adopts `architecture`, `blocks`, `filters` and `separation` from the
   weights it names**, where the specification stayed silent - the mechanism the semantic
   engine has had since `0.8.0a13` and the detection engine since `0.3.0a12`, now with the
   one entry this task cannot be read without.

   `separation` is that entry and it is the first adopted value that is not a shape:
   `dsbowl-instances` is a rim at **width 2**, which is a training decision rather than a
   geometry, and no reader can guess it. It is adopted as a block built from the sidecar's
   `separation_settings` and then re-validated with the rest of the model, so an adopted
   value faces every check a written one does. The engine **prints what it took** - the
   argument for adopting a training decision at all is that the reader still sees it, and
   the run record carries it as if written.

   What this cost a reader before, measured by writing R's instance vignette: the shortest
   useful thing anybody could do with the published weights named four of this model's
   fields, and the only place three of them are written down is a model card.

        u_net("objects", input_shape = c(256, 256), blocks = 4, filters = 16,
              separation = separation_rim(width = 2),
              weights = "dsbowl-instances", fit = FALSE)

   is now

        u_net("objects", input_shape = c(256, 256),
              weights = "dsbowl-instances", fit = FALSE)

 - **`separation_settings` is written by the engine** into every instance weights file it
   exports. `dsbowl-instances` carries it because its publishing script added it by hand,
   so an adoption reading that key would have worked for exactly one file in the world.
   It stays out of the fingerprint's comparison on purpose: a 2-pixel rim is both the right
   shape and the right meaning for a model asking for 3, which is the ordinary way to start
   one run from another.

 - **`classes_from`: one binary mask file per class**, the fourth way of saying what a
   mask's classes are and the only one that can express classes which overlap
   *independently* - a vessel crossing an organ, or two annotators' masks of the same
   structure. One pattern per class, in class order, matched against a sample's mask files
   through the same matcher `channels_from` uses: exactly one file per pattern, no file
   matched twice, no file left out. `pyplatypus#91`, which has been open since 2022 and is
   this half of it; the author's decision of 2026-10-12 was *nested regions first*, and
   deferred this rather than dropping it.

   **The whole feature is a reader.** Everything from the target onwards arrived with
   `regions` in `0.8.0a14` and reads `data.multilabel`, which this naming now also answers:
   a sigmoid per channel, the binary forms of `cce` and `focal`, metrics thresholding each
   channel, a prediction that is a stack, `save_masks` refusing an overlapping prediction
   and `save_class_masks` writing one file per class. What was missing is the step before
   it. `unite_masks` takes the element-wise maximum of a sample's mask files - the Data
   Science Bowl ships one per nucleus - and **that maximum is exactly where per-class
   identity is destroyed**: two overlapping annotations become one blob and nothing
   downstream can tell there were two. The test that carries this release is the control for
   it: the same two files, united, are one blob with two values; coded, they are two classes
   and their overlap.

   **One integer per voxel, still.** The coded form sets a bit per class, so tiling and
   augmentation are untouched - both operate on a single-channel array - and the expansion
   to channels happens in the last step before the loss, where `np.eye(n_class)[classes]`
   and the regions' table already live. This is the third expansion and the only one that is
   arithmetic rather than a table: a table of *combinations* would need two to the power of
   the classes, where the regions' table needs one row per label value. Lossless to 62
   classes in an int64 and refused above that rather than wrapping into a class nobody asked
   for.

   The threshold for *in this class* is **above zero rather than a midpoint**, which is the
   decision in it worth not re-taking: these files arrive as both 0/255 pictures and 0/1
   label maps, and `>= 128` reads every voxel of the second kind as background - a whole
   dataset silently empty, scoring a Dice of 0 with nothing to say why.

 - **`_class_fingerprint` records the patterns**, the way detection records its class
   *names*: two independent classes read from two files and two exclusive ones read from a
   colormap are the same count and the same head, so weights from one read back as the other
   would label every channel wrongly and answer plausibly. `n_class` alone cannot tell the
   four namings apart, which is why the naming itself goes into every run record and every
   weights sidecar.

### Fixed

 - **`InstanceEngine._load_weights` explained the absence of all this by a fact that had
   stopped being true.** Its docstring said *there are no published instance weights, so
   there is no registry entry to read `blocks` and `filters` out of* - true when it was
   written, and ended by `dsbowl-instances` one release later without anybody noticing. A
   declaration beside a mechanism that no longer describes it, which is the same shape as
   `#230` and `#231`, and found the same way: by writing the documentation a reader would
   follow.

### Changed

 - **`separation` is no longer required outright**: it may be omitted when `weights` names
   a file that records it, and that permission is paid for in two places rather than
   assumed in one. The specification refuses the omission when there are no weights at all,
   which it can check without resolving anything; the engine refuses a file whose sidecar
   does not record the separation, which needs the registry and therefore cannot live in a
   layer that must be validatable without a download. In the window between the two, asking
   a model what its head predicts answers with a sentence naming the file rather than with
   an attribute error on a field nobody wrote.

 - **`decode` is deliberately not adoptable.** It is the reader's question rather than the
   file's property: one trained file answers it three ways - 0.5445, 0.5390 and 0.4822 on
   the published instance weights - and the sidecar lists every decoding it was measured
   with, so adopting from that list would silently pick whichever name came first.

 - **One adoption mechanism instead of two copies of it.** `_adopt_from_weights` existed
   twice, character for character apart from a type hint, in `Engine` and `DetectionEngine`;
   this release needed a third with one asymmetric entry. It lives on `EngineBase` now and
   each engine declares only `ADOPTABLE`, which became a mapping of *field to the key the
   sidecar records it under* - the two names differ exactly once, and that is the whole
   reason for the shape.

 - **One matcher, two nouns.** `match_channels` takes `what=` and says "class" where a class
   is what it matched. Every guard is the same for both fields and the only difference is
   whether a mismatch mislabels a measurement or a class; a second copy with the other noun
   is the shape §3F of the workbench's notes is a list of.

 - **Both multilabel namings are named by the refusals that mention one.** `lovasz` and a
   written `include_background: false` are refused for `classes_from` exactly as for
   `regions`, and the message now names whichever field was written - a refusal that says
   `regions` to somebody who wrote `classes_from` sends them looking for a field they have
   not got. Same for the warning about unmatched voxels, which stays suppressed: neither
   naming has a background channel, so on a small target the fraction sits near 1 and is the
   correct answer.

 - **One place reads a sample's masks**, instead of the copy `_load` and `inspect_masks` each
   had. The third naming would have made it three, and the read is not where either of them
   has anything of its own to say - which is also what gives `inspect_masks` its answer for
   free here: a pattern matching a file that is empty in every sample read is a channel of
   zeros, and `missing_classes` reports it by class with the pattern quoted.

### Still open

 - **The R half.** `segmentation_data()` enumerates its parameters, so `classes_from` is an
   `unused argument` there today, and R's exclusivity check has to become four-way - which
   is a code change rather than a pin bump, measured on `regions` and recorded on
   `rplatypus#152`. Engine first, published, then R: `pyplatypus#91` stays open for it, and
   this release does not pretend otherwise.

## [0.8.0a27] - 2026-10-16

One line, and it is a defect in `0.8.0a26` found by answering a question about it rather
than by a test. A **flag on the metric** replaces a list of names, and the two layers that
had the answer separately now read one.

### Fixed

 - **A tiled run accepted a callback watching `val_surface_dice`, which its epochs cannot
   produce.** The trainer decides what an epoch reports by asking the built metric
   (`metric.accumulates`); the specification decided what a callback may watch from
   `_WHOLE_MASK_METRICS = frozenset({"cldice"})`, a list of names in `spec/models.py`. So
   `surface_dice` arrived in `0.8.0a26` on one side of the question and not the other, and
   `monitor: val_surface_dice` beside `splits` was accepted, trained for an epoch and then
   raised `KeyError` out of `state.logs` - which `EarlyStopping` indexes directly - with
   the epoch already paid for.

   The fix is not the missing name. `whole_mask` is now a class flag on the metric
   component, the shape `continuous` and `supports_multilabel` already have on a loss, and
   `monitorable` asks the component instead of a list. It stays in the specification layer
   and imports nothing, so `tests/spec/test_spec_layer_is_light.py` is unmoved: the reason
   the list existed was to keep torch out of the spec, and a flag on the spec's own class
   does that without being a second place for the answer to live.

   **A test pairs the two answers**, which is the part that outlives this release: for every
   metric the configuration offers, the component's `whole_mask` must be the negation of the
   built metric's `accumulates`, with a control asserting both values actually occur - a
   pairing is no evidence in a package where every metric answers the same way. The three
   tests that cover the tiled case were each written for `cldice` and are now parametrised
   over the metrics that *say* they are whole-mask, so the next one is covered by arriving
   rather than by somebody remembering three files. §4's passing test whose fixture never
   contained the case, twice in two releases.

   Nothing else changes: `surface_dice` and `cldice` were always measured per case by
   `evaluate_cases` on a tiled run and always absent from its epoch history, which is what
   both docstrings said while the specification disagreed.

## [0.8.0a26] - 2026-10-15

Two of the four issues in the "Losses and metrics" milestone, and **the other two are
answered rather than built** - both of them were filed with the measurement that says not
to. `maju116/rplatypus#113` and `#114`.

### Added

 - **`surface_dice`, a metric: how much of each boundary lies within a tolerance of the
   other.** `rplatypus#113`. The third question this package can ask of a mask, after
   volume (`dice`) and connectedness (`cldice`): **how far wrong** is the edge. The volume
   vignette ended on a model at Dice 0.88 whose volumes were 12% to 23% too large, and this
   is the number that says so directly rather than being inferred from two columns that
   disagree - and it is what `boundary` optimises while nothing here reported it, so that
   loss was being judged by the metric it exists to improve on.

   Normalised surface Dice as published (Nikolov et al., 2018), symmetric: the truth's
   surface near the prediction is the boundary's recall, the prediction's near the truth is
   its precision, and one ratio weights them by surface size.

   **`tau` is required and has no default**, because a tolerance belongs to the structure
   and the dataset - conventionally the inter-observer variability for that structure. It is
   in millimetres when `target_spacing` puts the samples on a common grid and in voxels when
   it does not, which is the same pair of answers the signed distance map gives and for the
   same reason. **The only metric here that reads the spacing**, which is why it belongs in
   a package that carries one: a 0.5 mm and a 2.5 mm acquisition are not comparable in
   voxels at all. The spacing reaches it the way `multilabel` reaches the others - the
   builder joins the spec's tolerance to the data's voxel - and `InstanceData` has no
   `target_spacing` field, so an instance run measures in voxels and says so.

   **Written as a dilation rather than a distance transform, and measured.** A voxel is
   within `tau` of a surface exactly when a ball of radius `tau` centred on it contains a
   surface voxel, so the question is a morphological dilation by that ball - which costs the
   ball's volume per voxel where a transform costs the whole image whatever `tau` is:

        256x256, batch 8, two channels     2.7 ms on the GPU      41 ms on the processor
        128^3,            two channels    23.8 ms on the GPU     372 ms on the processor
        the same by scipy's transform                            948 ms on the processor

   `#113` worried that a distance transform at 128^3 costs 710 ms, which is the whole epoch
   budget; this is thirty times less and can therefore run every batch. Verified against
   `scipy.ndimage.distance_transform_edt` - **a different algorithm**, so agreement is
   evidence rather than a shared bug - exactly at four tolerances over isotropic and
   anisotropic spacings at both ranks, and the whole metric against a scipy-only
   implementation to 1e-6 over thirty pairs.

   **`include_background` defaults to False here**, and for a measured reason rather than
   the usual one: the background's boundary is the foreground's boundary *plus the edge of
   the picture*, and that edge agrees between any two masks whatever the model did.
   Measured on four random pairs at `tau` 2, background 0.6110-0.7030 against foreground
   0.5583-0.6266 - higher every time. That is agreement on the field of view. For the same
   reason a mask running off the edge has no surface there, which also makes two solid
   masks score 1.0 rather than 0/0.

 - **`generalized_dice`, a loss: classes weighted by the inverse square of their volume.**
   `rplatypus#114`, in both `loss` and `boundary`'s `region`.

   **The usual argument for it does not hold in this package and the help page says so.**
   Generalized Dice is introduced as a fix for Dice being dominated by the background;
   `dice` here is a mean over the class channels, so a lesion in 1% of a frame already
   carries the same weight as the 99% around it. Measured on a 1% target with two
   predictions - one finding the lesion and leaking a false positive of equal size, one
   missing it entirely and getting the background perfect:

        dice               0.1658 finding   0.4731 missing   prefers finding by 0.3073
        cce                0.1600 finding   0.1600 missing   indifferent
        cce_dice           0.1629 finding   0.3166 missing   prefers finding by 0.1536
        generalized_dice   0.3278 finding   0.9511 missing   prefers finding by 0.6233

   So what it adds is weight **beyond** an equal share - 1 to 9801 on that target rather
   than 1 to 1, and twice the margin between finding the lesion and missing it. And the
   objective that is genuinely dominated is the per-pixel half of `cce_dice`: to `cce` a
   false positive and a missed lesion of the same size are the same number of wrong pixels,
   which is the indifference in that table. **The claim in the issue was inherited from the
   literature and is wrong about this implementation**; it is corrected rather than copied,
   and the table is held by a test.

   Two decisions, both in the docstring and both tested. An **absent class** has no volume
   and so an infinite weight; it is given the largest finite weight in its sample, as the
   published implementations do, so inventing a class that is not in the picture costs as
   much as missing the rarest class that is - weight zero would make it free. A sample with
   **nothing present at all** has no finite weight to borrow, so its weights are flat and it
   is scored exactly as `dice` scores it; without that fallback every empty-truth sample
   reads a perfect 1.0 and a model predicting a lesion on a clean frame is never told.

   And **the weights are rescaled to a mean of one**, which the published quantity is
   invariant to and which is what makes `smooth` mean here what it means everywhere else:
   unscaled, the weights are of order 1/volume^2, so a `smooth` of 1.0 is not a small
   regulariser but the whole numerator. The control caught it - with three equal classes the
   coefficient read 0.855 where pooled Dice reads 0.355. With the rescaling, **flat weights
   are exactly Dice pooled over the classes**, which is a test rather than a hope.

 - **Where each may be named, asked of the configuration rather than of the union.** A loss
   in the union that every spec refuses is a name in a listing and nothing else, so this is
   tested through `from_dict`: `generalized_dice` is accepted as a semantic model's loss, as
   an instance model's loss for the mask, and as `boundary`'s region term - the combination
   worth having, since the case that motivates this loss and the case that motivates that
   one are the same picture. A flow block refuses it **by name**, because a flow field is a
   vector per pixel and this scores classes, and a detector refuses it for the reason it
   refuses every loss. `surface_dice` is a metric at both segmentation tasks and
   `object_detection` lists no metrics at all.

### Changed

 - **`available_losses()` is 11 and 14 where it was 10 and 13, and `available_metrics()` is
   5.** Nothing was edited to make that happen - both read the pydantic unions, which is
   what that design is for - but six pinned counts failed and each one is a place somebody
   had to look: the union's member count beside the two loader flags (looked: a region loss
   on one-hot classes, so neither flag), the semantic set written out by hand, two task
   listings, the component count in the description test, and two doctests. The R package's
   own listing test pins 13 and will fail when the pin moves, which is the ratchet working.

### Not built, and why - the other half of the milestone

 - **`cbDice` (`rplatypus#115`) stays a question.** The issue blocks it on a measurement and
   says so in as many words: *"if clDice is high while Dice is low - connected but the wrong
   calibre - that is this issue's evidence, and without it this stays open."* The measurement
   exists now and points the other way: on FIVES, 200 held-out images, Dice 0.8789 against
   clDice 0.8330 - the skeletons agree **less** than the volumes, which is breaks in thin
   vessels rather than the wrong calibre. Building it would be building for a problem nobody
   here has demonstrated.
 - **`DIoU`, `CIoU` and `EIoU` (`rplatypus#116`) stay unbuilt**, which is what that issue was
   filed to record. `giou` already lost to `offsets` on this data - per-seed spread 0.000116
   against 0.006863, sixty times tighter on the exact quantity GIoU targets - and
   `anchor_mean_iou` was 0.877 in all six runs, so the regime the whole family exists for is
   never entered. Three more names meaning roughly the same thing is the argument that kept
   the weights registry to one entry when four architectures scored within 0.0030.

## [0.8.0a25] - 2026-10-15

### Added

 - **`dsbowl-instances`, the fourth published weights, and the first whose score is not one
   number.** A U-Net trained to tell touching nuclei apart on BBBC038v1 - interior, rim and
   background, `separation: rim` at width 2 - pinned in the registry at `fd7bdc86`.
   `weights: dsbowl-instances` fetches it, the sidecar refuses a model it does not belong to,
   and nothing about the numbers can change under the name.

   Measured on the same 134 held-out images `dsbowl-unet` reports, seed 0, 120 epochs, 102
   minutes on one GTX 1070. Instance average precision, one score per image, and the mask all
   three rows share:

        nearest_seed           0.5445   sd 0.2554   5378 objects found of 5999
        watershed              0.5390   sd 0.2532   5378 objects found of 5999
        connected_components   0.4822   sd 0.2440   4914 objects found of 5999

        foreground dice        0.9118

   **One file, three scores, and the registry description is written so that neither can be
   quoted without the other.** The network predicts channels; turning them into objects is a
   separate step and `decode` is a list, so `evaluate()` returns a row per decoding on one
   forward pass - a second decoding of these 134 images costs 1.41 s against the 102 minutes.
   The spread between the best and the worst row is 0.0623, about seven times this
   configuration's run-to-run spread of 0.009, so a headline number without its decoding would
   be a number about whichever decoding happened to run. A test holds the description to its
   own arithmetic: every score in it names its decoding, the signal's claimed worth is the gap
   between the rows quoted, and the perfect-mask figure beside it is the larger of the two.

   **What the entry says about the signal is the milestone's finding and not a sales line.**
   `connected_components` ignores the separation signal entirely, so the signal is worth
   +0.0623 here, while substituting a perfect semantic mask into the same route is worth
   +0.2398 and a perfect mask through connected components caps *any* model at 0.7655. The
   mask is the limit on this data by nearly four to one. Published anyway, because the
   alternative is that the measurement exists only in a closed milestone and the weights it
   was taken on cannot be loaded by anyone who wants to argue with it.

 - **`examples/publish_dsbowl_instance_weights.py`**, which produced them. It reads its recipe
   out of `examples/data_science_bowl_instances.yaml` and the three recorded scores out of
   `#215`'s table rather than restating either, selects the one model through the same helper
   `compare_instance_separations.py` uses, and uploads nothing.

   **Its publication gate is a two-sided band**, calibrated at both ends: every decoding has
   to land within +/-0.027 of the recorded value - three times the 0.009 spread - and a run
   far *above* the band is refused as well, because that is a changed recipe rather than a
   better model. The refusal is diagnosable: the measurements are written before the band is
   read, so a refused run says which decoding missed by how much. Its positive control is a
   real refusal - a three-epoch run of the same script named all three gaps and wrote nothing.

   Four more controls, each one proven able to refuse:

   - **The split is checked by the number a different 134 could not produce**, 5999
     ground-truth objects, not by its size. 134 directories is the weak half of that
     fingerprint and `dsbowl-unet`'s card is why: the 134 on this disk were once a
     *different* 134 and the discrepancy was 0.01 of Dice.
   - **The data is fetched from the Broad rather than read off this disk**, and the local
     Kaggle copy was proven byte-identical to it first - 670 cases, images and per-object
     masks, sha256 - so the licence chain runs to BBBC038v1's CC0 rather than through a
     competition's terms.
   - **One model decoded three ways reports one mask**, asserted rather than assumed, because
     a table whose three rows disagree about `foreground_dice` is a table of three models.
   - **The sidecar cannot shadow what the weights say they are for.** The extra carrying the
     separation's settings is named `separation_settings`, and a check derives the reserved
     names from the spec's own fingerprint and refuses a collision. Named `separation` it
     would have overwritten the fingerprint's `"rim"` with a dict and made every later load
     of the file refuse it - caught before the training run rather than by a load failure
     afterwards.

 - **The published name was loaded the way a user reaches it, before this entry was written.**
   `weights: dsbowl-instances` with `fit: false`, resolved through the registry, downloaded
   from the pinned commit into an `HF_HOME` that had never seen it, sidecar checked against a
   model built from the specification, and scored on the same 134 images: all three decodings,
   the object counts and the foreground Dice came back **identical to seven decimals**, the
   largest disagreement 0.00e+00. The same file into a model with `filters: 32` or `blocks: 3`
   is refused by field name, which is the control that the acceptance means something.

 - **The Hub card gains its section and two older lines are corrected.** One card holds one
   section per model; this one is generated from the run's own measurements and a check holds
   every figure in it to the sidecar beside the weights, with the same check aimed at
   `dsbowl-unet`'s section as its control. The corrections are the R example's
   `library(platypus)`, which has been the wrong name since the package was renamed on
   2026-10-14 and was the one line a reader would have copied and had fail, and a `tags:` entry
   for what this file actually does.

 - **What R needs from this is the pin and nothing else.** `available_weights()` reads the
   registry from the engine, so a fourth published name appears on the R side as soon as the
   pin moves - the cheapest kind of R half there is, and the whole value of this release to an
   R user arrives through it.

## [0.8.0a24] - 2026-10-15

A declaration beside a mechanism that does not describe it, in the one place this drawing
exists for. `_colour_index`'s docstring said the wrap past the end of the palette "is
**said** rather than hidden - `overlay_instances` reports it", and `overlay_instances`
reported nothing: `grep -n warn pyplatypus/plots.py` returned no hit at all. The sentence
was right about what should happen, which is why this is a warning and not a deletion.

### Fixed

 - **`overlay_instances()` now warns when the picture needs more colours than the palette
   has**, naming how many it needs, how many there are and how many objects are drawn in a
   neighbour's colour. `maju116/pyplatypus#231`.

   The wrap is the one case this figure answers its own question wrongly in. Its subject is
   *where does one object end*, every other colour repeat is deliberate - a colour may
   repeat across the picture as long as it never repeats between neighbours - and a wrap is
   exactly the case where it does repeat between neighbours. **Provably, rather than
   usually:** a slot is only past the end of the palette once every colour before it was
   taken by a neighbour, so the fold lands on one a neighbour already holds and never on a
   free colour. The warning says that to the reader, so a test measures it rather than
   leaving the claim to the docstring that made this defect.

   Not reachable on anything measured here - the hardest of 670 Data Science Bowl cases
   needs 3 of the 8 default colours, with 6 neighbours for the worst single object - which
   is exactly why nothing noticed the gap for as long as the function has existed. A guard
   for a picture nobody here has is still worth having when the alternative is a figure that
   draws two objects as one and says nothing.

   Two things found while writing it, neither of them the warning:

   - **The fold is inside the greedy and that is not cosmetic.** Hoisting it out, so the
     colouring is computed in true slots and folded at the end, reads cleaner and changes
     the picture: measured on 440 overlays over 55 label maps, **234 of them differ**.
     Folding in the loop means a later neighbour avoids the colour an object is actually
     *drawn* in, so one wrap does not cascade into the rest of the picture. The R package
     folds inside the loop too, so the clean-looking version would also have been a
     divergence between the two. Found by a control recorded before the change rather than
     by reading the diff - and the first attempt at this fix was that version.
   - **A `Warns:` section cannot be published from here**, which is why nothing in this
     package had one and why the warning is documented as prose. `quartodoc` has no
     renderer for it: `uv run quartodoc build` raises `NotImplementedError: Unsupported
     type: DocstringSectionWarns`, so the `docs` job fails and the site cannot be built.
     Measured both ways - the section fails and the prose builds - because the four gates
     here do not include the reference build, and the absence of any `Warns:` in the
     package read as "no convention" when it was "no renderer".
   - **`_colour_index` cannot report what it was said to report**, and now does not claim
     to: it returns which objects wrapped, and `overlay_instances` - the function a person
     called, and the only one that knows the palette it was given - is what warns. The
     requirement in the message comes from the same function asked for a colouring with no
     palette at all, which is the number a short palette needs and cannot be read off a
     folded run.

### Changed

 - **`overlay_instances`, `overlay_instance_agreement` and `flow_to_colours` are now pairs
   rather than three functions with the same name on two sides.** The R package drew neither until
   `maju116/rplatypus#154`; it draws both now, in its own code, because drawing there
   touches no Python on purpose. So `tests/test_reference_groups.py` pins where both are
   filed, which is this side's half of §3F's pair - the other half is in R's
   `test-reference-groups.R`.

   Nothing in the wheel changes for this one and it needed no release of its own; it rides
   with `0.8.0a24`. Worth an entry anyway, because the
   subsequence test cannot catch what this one does: a name **absent** on one side is not a
   name filed differently, so until R had them there was nothing to disagree with and the
   pair could not have existed.

   What holds the *rule* rather than the filing is on R's side alone and has to be: it draws
   the same pictures with both packages and compares them pixel for pixel. The palette was
   already mirrored, so the two could have agreed about which eight colours exist while
   disagreeing about which object gets which. See `maju116/rplatypus#154`.

## [0.8.0a23] - 2026-10-14

A writer the package had promised twice and never had, and the test that finds the next
promise of that kind. `onehot_to_colours` has refused an overlapping prediction since
`regions` shipped in `0.8.0a14`, and its refusal ends *"overlapping classes are written one
file per class, not as a colour map"* - nine releases in which nothing wrote that file.

### Added

 - **`save_class_masks()` writes one binary picture per class**, which is what that refusal
   points at and what `Region.name`'s own `description=` has been naming since `0.8.0a14`.
   `<case><suffix>_<region>.png`, mode `L`, 0 or 255, in channel order - which is the order
   the regions were written in and the order every per-class score is reported in.

   Four decisions in it, each one a failure this project has already made once in another
   shape:

    - **A region's name is not a filename.** `whole tumour` becomes `whole_tumour` through
      `class_filename`, and two names that become the **same** filename are refused rather
      than written: one file would hold the second channel, the first would be silently
      gone, and the number of returned paths would still be right. That is the quiet shape,
      so it is the one that is refused by name.
    - **The name is not lowercased.** `Core` and `core` are one file on one platform and two
      on another, and the package already has one case-sensitivity story without a second.
    - **A channel nothing reaches is an empty picture, not a missing file.** A missing file
      reads as *the run did not produce that class*; an empty one reads as *it did and the
      answer was nothing*. Those are different findings and only one of them is true.
    - **A volume stack is refused.** A PNG carries neither voxel spacing nor orientation, so
      a volume written as pictures cannot be laid back over the scan it came from. Only rank
      5 is refused, because a single volume `(d, h, w, c)` and a stack of images have the
      same rank and nothing in the array tells them apart - the same limit `save_masks` has
      on the R side, and it is stated rather than guessed at.

   `threshold` defaults to `0.5`, which is what every metric in the package thresholds a
   multilabel channel at. A different number writes a picture that disagrees with the score
   printed beside it, and the docstring says so.

   The class names arrive as `classes=` rather than `names=`, decided by §3A before the
   function was released: R's `save_masks` and `save_volumes` already use `names` for the
   **files** they write, and a sibling writer where the same word means the classes is the
   quiet kind - a call copied from one to the other would pass filenames as class names, the
   counts would match whenever they happened to, and nothing would refuse it.

   Filed under **Masks and volumes** in the reference, which is the group R files
   `save_masks` and `save_volumes` under and which held nothing here until now. The shared
   order already carried the title; `tests/test_reference_groups.py` now pins the name's
   placement in both suites, which is the fifth such pair.

### Fixed

 - **The schema's prose named two functions that do not exist here, and now a test says it
   cannot.** `tests/test_schema_prose_names.py` reads every `description` out of
   `spec_schema()` - the strings `tools/build_config_reference.py` publishes as the
   configuration reference - and asserts every `` `name()` `` in them resolves to something
   this package has. It found the two it was written for and one it was not:

    - `save_class_masks`, named in `Region.name` and built above. The name was decided on
      `maju116/platypus#91` and written into the schema before anything existed; nothing
      read the sentence again for nine releases.
    - `training_history()`, in `CsvLogger.path` and its class docstring, which is the **R**
      package's function. True for the R reader it was written with in mind and a dead end
      for the Python reader the generated page is for. Both now say `fit()`, which is what
      this package answers the same question with.

   §4's *"an instruction addressed to a human is code nothing runs"*, in the one place this
   project generates documentation out of code. The test is the engine's because the schema
   and the page are the engine's, and it carries the usual guard against passing by looking
   at nothing - it asserts it found more than 150 descriptions and more than five names
   before asserting anything about them.

### Documentation

 - **The guide has a section on overlapping classes**, which `0.8.0a14` shipped the field for
   and deliberately did not document in narrative form: *Overlapping classes instead of one
   per voxel*, between the augmentation section and the instance one, because `regions` is
   still `semantic_segmentation` and what changes is what a channel means. The page's section
   list is pinned by a test here and the same list in R's suite, so **this lands in both
   repositories or neither**; `maju116/rplatypus#152` is the other half.

   The numbers in it are measured rather than asserted: the same masks are **four** classes
   written as `labels: [0, 1, 2, 4]` and **three** channels written as the three BraTS
   regions, and a voxel whose value is `4` belongs to all three at once. The YAML block is
   validated by `test_every_configuration_in_the_guide_is_accepted` like every other block on
   the page - checked by breaking it, which fails that test at its line rather than passing
   quietly.

   The section says one voxel gets one colour and a voxel here may be in three classes, and
   it now names what writes such a prediction out: `save_class_masks()`, added in this same
   release. It was written without that sentence, because the function existed in neither
   package at the time and the honest thing was to say nothing rather than to name it;
   `maju116/rplatypus#152` is where the decision to build it was taken.

### Changed

 - **`_brand.yml` now names the test that holds its two copies together**, because as of
   `maju116/rplatypus#160` there is one. It is on the R side alone - it fetches this file
   from this repository's default branch and compares it line by line - for the reason the
   field-prose ledger is one-directional: the engine owns the look, so R's copy is the copy.
   Nothing here changes but the comment, and nothing is in the wheel either way.

 - **The R package is `rplatypus`.** It was `platypus` until 2026-10-14; the name collided
   with `Platypus`, archived from CRAN on 2026-02-02, which archiving does not release, so
   it was renamed rather than waiting on a reserved name. One name per language, beside this
   package on PyPI. **Nothing in the wheel changes**: this is the engine's own links, its
   `_brand.yml`, and the `citation("rplatypus")` the generated Citation page prints -
   §3E's shape, a change R needs nothing from, said here rather than only in a commit
   message.

   The GitHub repository was renamed with it and the former addresses redirect, so the
   issue cross-references in the entries below are left exactly as they were written. They
   name `maju116/platypus` because that is what it was called when they were written, and
   they still resolve.

   `maju116/platypus-weights` is a **different repository** - the published weights on the
   Hub - and is untouched. Worth saying because a rename done by pattern would have taken
   it, and `resolve_weights` would then look for weights nobody published.

The second entry under this heading and the same reason as the first: it records an
**outcome**, and neither `examples/` nor the guide page is in the wheel. What *is* in the
wheel is one docstring, so nothing here changes what a run does. What is new is that the
outcome includes two corrections to things this project had written down - a column it
printed, and a claim about a table it builds.

### Measured

 - **Both separation signals, five decodings, one command** (#215, the instance milestone's
   deliverable). `examples/compare_instance_separations.py` against
   `examples/data_science_bowl_instances.yaml`, 185 minutes on one machine, scored on the
   134 validation cases `--expect-cases` pins by name:

        model  separation  decode                class dice  AP      objects  truth
        rim    rim         nearest_seed          0.8550      0.5442  5414     5999
        rim    rim         watershed             0.8550      0.5396  5414     5999
        rim    rim         connected_components  0.8550      0.4767  4875     5999
        flows  flows       follow_flows          0.9464      0.5174  5400     5999
        flows  flows       connected_components  0.9464      0.4512  4849     5999

   **The signal is worth +0.0676 to the rim route and +0.0661 to the flow route**, over each
   route's own connected-components row. Both reproduce what was measured separately -
   +0.0640 in `0.8.0a15`'s sweep and +0.0680 in #212 - so §4bq's reading stands: a perfect
   mask is worth +0.2398 and the mask is the limit by a factor of 3.6.

   The noise floor is measured here rather than borrowed. The flow model was configured
   identically in both runs of this file and its average precision moved **0.0112** between
   them, which is larger than the 0.0074 two runs of one configuration differed by in
   `0.8.0a15`'s sweep - so a gap smaller than 0.0112 says nothing on this table, and the rim
   route's 0.0268 lead over the flow route is 2.4x it on a sample of two runs.

 - **The rim model was undertrained at 60 epochs and is not undertrained at 120**, which is
   why the two models' epoch counts differ in the configuration and why that asymmetry is a
   comment rather than a typo. Doubling it bought +0.0451 of average precision; the last
   twenty epochs of the hundred and twenty bought +0.0018, and the best epoch was 118 of 120.

### Documentation

 - **The configuration guide has a section on the third task**, in both packages:
   *Instances instead of one mask*, between the model's own fields and detection. It is the
   §3C rule that a change to how a run is written belongs in the guide on both sides with the
   same sections, and `task: instance_segmentation` shipped in `0.8.0a20` without it - the
   guide had zero occurrences of the word.

   What a reader needs that no reference page tells them: the data block gets **shorter**
   (there is no `colormap` and no `labels`, because the classes are always background and
   object and the per-object files are what name the objects), `separation` is one block per
   model with no default while `decode` is a list, no value appears in both because one is a
   noun and the other a verb, and a model's own `loss` scores the mask while a `flows` block
   carries its own for the vector channels. The section also carries what the task is worth -
   +0.068 against a perfect mask's +0.2398 - for the same reason the reference does.

   Both guides' YAML is validated by each suite, and the new block is: it was mutated to name
   `follow_flows` under `separation: rim` and the Python test failed on that line.

### Fixed

 - **`evaluate()`'s own docstring claimed the yardstick is in every model's rows "whether it
   was asked for or not", and it is not.** `connected_components` is in the pair `decode`
   *derives* to when a model stayed silent; a model that writes a `decode` list gets the list
   it wrote and nothing added. The code is right - the engine does not add a row to somebody's
   table that they did not ask for - and three pieces of prose were wrong, including
   `0.8.0a20`'s own entry, which is corrected in place below with the correction left visible.

   **It went unnoticed because only the kind half had a test.**
   `test_the_yardstick_is_in_the_table_without_being_asked_for` covers the derived pair and
   says so in its docstring; nothing covered a written list, so the claim about the branch
   with no test is the one that drifted. `test_a_written_decode_list_is_what_you_get` is the
   missing half, and it fails when the code is mutated to make the old sentence true.

   `examples/compare_instance_separations.py` already kept the yardstick in its own `--decode`
   handling, which in hindsight is the evidence: an example compensating for behaviour the
   engine was documented as having.

 - **The table's `fg dice` column was neither a foreground Dice nor one quantity**, and is
   now called `class dice`. It is the engine's `dice` metric, a mean over the class channels
   the model's head actually has: three for `rim` - background, interior and the rim band -
   and two for `flows`, whose metrics are shown the class block alone. So the rim model's
   number is averaged against a thin band scoring 0.6857 where its own background and
   interior score 0.9852 and 0.8721, and reading it across the two models compares two
   different things.

   **Measured by identical code on both, on the same 134 cases: the binary foreground Dice
   is 0.9159 for the rim model and 0.9097 for the flow model.** Where the column read them
   0.09 apart it is 0.0062, and the sign is the other way round - the rim model has the
   better mask. The control is the Unreleased entry above this one: foreground Dice held at
   **0.917-0.920** across four rim runs there, which is the family 0.9159 belongs to and
   0.8565 does not.

   `examples/train_instance_separation.py:261` had already written the rule down - *"against
   three is not the comparison; foreground against foreground is"* - and computes the
   quantity by hand at line 263. So this is a lesson lost rather than one learnt, which is
   §4's "documentation rots in the direction of the work having been done" wearing a column
   heading. What #215 asked for, a foreground Dice the table can be read across, needs the
   engine to return one. **This entry said `evaluate()` has no such column; `0.8.0a22`
   below gives it one** (#222), and the example prints that rather than a third hand-rolled
   copy. The two training scripts keep theirs, which is a decision rather than an oversight:
   they score an empty case as 1.0 where the engine's column reports it as unscorable, so
   substituting silently would move a published number's definition.

 - `examples/data_science_bowl_instances.yaml` named its checkpoints `weights.safetensors`
   while `model_checkpoint` is `torch.save` - a pickle in a zip, whatever the path says.
   They are `weights.pt` now. `export_weights()` remains the safetensors path, and naming a
   pickle after the format this package chose *because* it is not a pickle is the kind of
   label that gets believed.

## [0.8.0a22] - 2026-10-14

One column, and it is the one an instance table is read across. `#222` came out of `#215`
and was deliberately left out of the instance milestone: the methods were delivered and the
measurement taken, and this is a defect in how a result is **reported**.

### Added

 - **`evaluate()` carries `foreground_dice` on every instance row** - the binary overlap of
   anything the model called an object against anything that is one, computed identically
   whatever the separation signal is. The table already had a mask column and it was not one
   quantity: `dice` reduces over the class channels a head has, three for `rim` (background,
   interior, rim band) against two for `flows`, so on the Data Science Bowl's 134 validation
   cases it showed the two masks **0.09 apart where they are 0.0062 apart, and with the sign
   reversed**. `class dice` is the honest name for that one and it stays, beside the new
   column rather than instead of it, because a rim model's three class numbers are where a
   thin band that never learned shows up at all.

   Three decisions, each written into the function's docstring because each is worth more
   than some of the effects this table is read for:

    - **Both sides binarised, through `read_prediction`.** That is the one place the channel
      layout is read - the vectors sit in front of the classes for `flows` and nowhere for
      `rim` - so the column cannot acquire a second opinion about where the classes start.
    - **Unsmoothed, and an unscorable case says so.** The `+1` the training metrics carry
      scores an empty prediction on empty truth 1.0; `foreground_dice` returns `None` there,
      which is this module's existing convention - `score_cases` leaves such a case's
      `average_precision` at `None` and `summarise_instances` filters on exactly that.
    - **Averaged over images**, which is how the `average_precision` beside it is pooled. The
      training metrics pool over *batches* with each batch weighed equally, worth 0.0088 on
      that split, so a row carrying both poolings would hide a difference larger than several
      it exists to show.

   **It costs no second forward pass.** The number is taken inside `_decoded`'s existing
   loop, once per case rather than once per decoding, which is also what makes "the same for
   every row of one model" structural: `_Decoded` carries the mask number beside the
   decodings instead of inside each of them.

 - `examples/compare_instance_separations.py` prints it as `fg dice`, from the engine's row.
   The issue asked for this rather than a third copy of the arithmetic, and
   `test_the_instance_table_does_not_call_its_dice_a_foreground_dice` still holds the *other*
   column's title, which is the label that was wrong in the published table.

### Fixed

 - `tests/test_instance_engine.py`'s module docstring said the yardstick is in every row
   "whether it was asked for or not" - the **fourth** copy of the claim `0.8.0a21`'s entry
   corrected in three places. Found by working in the file rather than by a sweep, which is
   the argument for the sweep.

### Documentation

 - **The guide says which mask column to read across models**, in the section the third task
   got in `0.8.0a21`'s cycle. The engine's copy ships here because it describes what this
   release returns; R's copy is written at the same time and ships with the pin move, because
   a vignette cannot describe a column its pinned engine does not return.

### The R half

Almost nothing, and structurally rather than by decision: R's score table *is* the engine's
rows, so the column appears on the R side the moment the pin moves - §3E's shape, the same
one `0.8.0a12` had, where a release R needs nothing from is still a release whose whole value
to an R user arrives through the pin. What R owes beyond the pin is one assertion, because a
column that arrives untested is the next thing to drift: `test-instances.R` now names
`foreground_dice` and holds it to one value per model.

## [0.8.0a21] - 2026-10-14

One refusal, moved to the class it belongs to. A **semantic** model's own `loss` accepted
`mse`, `l1` and `smooth_l1`, trained with them, and reported a small number for a mask that
was half missing (#223).

### Fixed

 - **`loss: mse` on a semantic model is refused, by name and before anything is built.** The
   validator existed and sat on `InstanceModel`, which is a *subclass* of
   `SegmentationModel`, while the thing it protects - `loss` scoring a target of classes -
   belongs to the parent. So the semantic task had no guard, and nothing downstream supplied
   one: `Engine` builds the loader with `with_foreground=loss_needs_mask(model.loss)` and the
   trainer hands a third tensor to any loss that asks for one, so a run completed.

   **The number it reported was the good one, which is why this is refused rather than
   documented.** `where` defaults to `foreground`, and for a semantic target the foreground
   is *not the background channel* - so the objective was evaluated only inside the labelled
   objects and the background was unconstrained. Measured on eight 32x32 samples whose single
   object is where the picture is bright, 80 epochs, `u_net`, a task `cce` solves exactly:

        the model's own loss                 val_dice   val_loss
        cce (control)                        1.0000     0.086507
        mse, silent - so where: foreground   0.5367     0.001430
        mse, where: everywhere               0.9838     0.055150

   The middle row is the finding: a validation loss sixty times smaller than the control's,
   and the only place the missing half of the mask shows is `val_dice`.

   The last row is why the refusal is a decision rather than a correction of nonsense, and it
   is recorded rather than buried: an unactivated squared error is a crude surrogate, not
   gibberish. It is refused anyway, because `loss` is the objective on the class channels and
   an activated regression would be a new loss with its own name and its own measurement -
   not this family, whose entire content is `where` and whose documentation is about flow
   fields. The decision is on #223 with the two alternatives it was taken against.

   Admissibility is still read off the `continuous` flag on the loss itself, so a fourth
   regression loss is refused with nothing edited. `InstanceModel` now contributes only the
   extra clause that names where a continuous loss *does* belong - the flow block's own
   `loss` - so that refusal still names both halves.

   **Nothing in this repository used it**: no example, configuration, test or fixture builds
   a semantic model with a regression loss, so the behaviour change costs nothing here. An R
   caller reaches the same hole through `loss_mse()`, which R gains when its pin moves.

### Changed

 - **`available_losses()` answers per task**: ten at `semantic_segmentation`, thirteen at
   `instance_segmentation`. The three in the difference are the regression family, and at the
   semantic task they can be named nowhere - a listing that offers what the next validator
   refuses is a list maintained against a validator, which is what `continuous` exists to
   avoid. The instance task keeps all thirteen because a `separation: flows` block's own
   `loss` is that same union; which loss goes where is the tree, and `available_separations()`
   is where the tree lives.

   The listing-against-schema test now asks the losses of the instance task, because that is
   the only task whose answer can equal the schema's `discriminator.mapping`: the mapping is
   per *field* and cannot express a validator. The semantic task's shorter list has its own
   test which names the three excluded losses literally rather than deriving them the way the
   code does - so a fourth regression loss fails it, and failing is right.

   R reads this listing through the bridge, so `available_losses()` there changes with the
   pin and the R suite's count moves with it. Said here rather than discovered: §3E.

## [0.8.0a20] - 2026-10-13

`task: instance_segmentation`: the third task, and the last engine piece of the instance
work (#160). Two axes on a model - `separation`, one per model, and `decode`, a list - a
head whose width is derived rather than stated, and a score computed after a post-process
rather than from `(logits, target)`.

**What the tag is honestly worth is in its own documentation and it is not flattering.** On
the Data Science Bowl the separation signal is worth +0.068 of instance average precision at
best while a perfect mask is worth +0.2398, so the mask is the limit by a factor of 3.6.
`connected_components` - which ignores the signal entirely - is in the pair `decode` derives
to when a model stayed silent, because it is the row the signal's row has to be read against,
and a *perfect* semantic mask scores 0.7655 through it. (This sentence read "in every model's
decodings whether it was asked for or not" until the Unreleased entry above corrected it: that
was never true of a model which wrote its own `decode` list.) The capability is real, the
ceiling is measured, and both are in the configuration reference rather than in a footnote.

### Added

- `task: instance_segmentation`, with `InstanceData`, `InstanceModel` and `InstanceSpec`.
  Two new fields and everything else inherited: `input_shape`, `channels`, `blocks`,
  `filters`, `batch_size`, `epochs`, `splits`, the pretrained encoders, the callbacks and
  the augmentation all mean what they mean for a semantic model.
- `separation:`, **a block named `rim` or `flows`** carrying that mechanism's own settings,
  which sets the head's width and which losses are admissible. **`hv` is deliberately not a
  name**: its markers lose to a one-pixel erosion on perfect inputs (0.9382 against 0.9807),
  so naming it would promise a mechanism #213 measured as absent.
- `decode:`, a list of blocks keyed on `name` - `connected_components`, `nearest_seed`,
  `watershed`, `follow_flows` - with each block declaring which separation it can read, so
  `follow_flows` with `separation: {name: rim}` is refused at load time with both halves
  named. Left unset it derives to the signal's own decoding plus the yardstick.
- The mechanisms' own settings, **inside the block they belong to**: `width` under `rim`,
  and `mode`, `loss` and `weight` under `flows`. So a setting crossed with the wrong
  mechanism is not a refusal somebody had to write, it is an unknown key - the first shape
  of this field was `separation: rim` beside `rim_width`, `flow_mode`, `flow_loss` and
  `flow_weight`, which parses on every model and therefore needed a table of which field
  belongs to which mechanism and a validator reading it. `extra="forbid"` was already doing
  that job for every other key in a specification.
  `mode` defaults to `diffusion` on a measurement (0.9924 decoded against `centroid`'s
  0.9478); **`width` defaults to 2 on an argument rather than a measurement**, and the field
  says so - the published sweep peaked at 3 pixels of *source* resolution, which on a
  520-pixel case read at 256 is about 1.5 pixels of model input, and a sweep in model pixels
  has not been run.
- `InstanceEngine`, on `EngineBase`: `fit`, `evaluate`, `evaluate_cases`, `predict`,
  `predict_instances`, `export_weights`, `best_model`, `dataset`, `loader`. `evaluate()`
  returns one row per (model, decoding); `evaluate_cases()` is one row per case for one
  decoding, so `summarise_cases` keeps meaning what it means.
- `InstanceDataset`, which reads the per-object files as a label map instead of uniting
  them, and builds the target **after** the resize, the tiling and the augmentation -
  `pyplatypus#210`'s rule, pinned here by a recording augmenter.
- `pyplatypus.instances.targets` and `pyplatypus.instances.decode`: the channel contract
  going in and coming back, with a test that a perfect prediction decodes into the objects
  it was built from. That is the only check that catches a channel swapped between the two,
  because both arrays keep the right shape and range.
- `available_decodings()`, with a `needs` column read off each decoding rather than written
  down; `available_separations()`, with the head width each signal needs derived rather than
  stated; and a third row in `available_tasks()`.
- `examples/data_science_bowl_instances.yaml` and
  `examples/compare_instance_separations.py`: both signals and five decodings from one
  configuration file. **This is the test of whether the tag earned its place** - the script
  says where the data is and the engine does the rest.
- A refusal specific to this task: a split whose objects have all vanished in the resize.
  The objects are on disk, the resize removes them, and without the check the loss falls,
  every metric stays finite and the run learns to answer "nothing here". Some cases losing
  theirs is a warning with the count; all of them is an error naming `input_shape`.

### Changed

- `Trainer.activate` is a method rather than one inlined call, which is what `InstanceTrainer`
  overrides: a flows head is two raw vector channels in front of a block of classes, and a
  softmax across all of them would return a number, a gradient and no error anywhere.
- A dataset answers `foreground_of` for its own channels. The loader used to call
  `onehot_to_foreground`, which reads channel 0 as background - for a flows target that is
  the row component of a vector field, and the loss would have been handed a mask of
  everything.
- `labels_in_test_split` is one function the two mask-reading engines share, rather than the
  same method written twice. `EngineBase` diverged twice before, and the second time the fix
  was written twice on the same night.
- `tests/test_engine_base.py` has a third column rather than a second file: the thing worth
  asserting is that the three engines give *the same* answer, which a file per engine cannot
  say.
- The two hand-assembled instance scripts say what they now are - the record of the
  measurements they produced, not the way to run one - and the sentence claiming the engine
  could not derive a flow head's width is corrected rather than quietly deleted.

### Fixed

- The configuration reference published the third task's `data:` and `models:` under the
  heading "for detection". `short = "segmentation" if tag == "semantic_segmentation" else
  "detection"` was a binary over a set that grew, and the heading-collision test is what
  caught it rather than anybody reading the page.

## [0.8.0a19] - 2026-10-12

Looking at instance results instead of reading their arrays (#214), the trained half of
`flows` (#212), and `#213`'s cheap test answered before anything was trained for it. All
three end in the same place, which is the entry's finding rather than three coincidences:
the separation signal is not what limits instance segmentation on this data. The mask is.

### Added

 - **`overlay_instances()`, `overlay_instance_agreement()` and `flow_to_colours()`** in
   `pyplatypus.plots`. An instance result has no figure today - `overlay_mask` paints every
   object one colour, which is the one thing an instance map must not do - and `#159`'s four
   training runs were read entirely off tables.

   **`overlay_instances` colours an exact graph colouring and that was an open question on
   `#214`, settled by measurement rather than by preference.** What a reader needs from an
   instance map is *where one object ends and the next begins*, which is **adjacency, not
   identity**, so a colour may repeat as long as it never repeats between neighbours.
   Measured over all 670 Data Science Bowl cases, 524 of which hold touching objects and
   **5436 adjacent pairs** between them: an exact colouring (Welsh-Powell) needs **at most
   3 of the 8 colours `CLASS_COLOURS` already has** - 146 cases need one, 412 two, 112
   three - against at most 6 neighbours for any one object. The cheap alternative the issue
   offered, cycling the palette by object value, **paints 550 of those 5436 pairs in one
   colour: 10.12%**. One touching pair in ten drawn as one object, in a figure whose whole
   subject is whether two objects are two.

   **`overlay_instance_agreement` matches objects with `iou_matrix`, the function the metric
   uses, and at the same strictness** - which is a correction rather than a design note. It
   matched at `iou >= threshold` for an hour; `average_precision` compares strictly greater,
   and a merged pair sitting over two equal truth objects lands at exactly 0.5, so the
   figure would have drawn one hit where the metric counts two misses. Caught by a test
   written against what the figure *means* rather than what it draws, and pinned by one.
   The threshold below 0.5 that the metric refuses is refused here too, in the metric's own
   words.

   **Nothing was added to `pyplatypus.style`**, which is why the R pin does not move for
   this (`§3E`): hue-for-direction is the convention optical flow has used for twenty years
   and not a decision this project gets to make differently, and the drawing style R reads
   from the engine is unchanged. `platypus#154` carries the R half.

 - **`examples/train_instance_flows.py`**, the second half of `#212`: the flow target
   trained and scored, with the pair of oracles that says which half of the pipeline the
   result is limited by. Sixty epochs on `dsbowl-unet`'s architecture and the 536/134
   directory split, the target rebuilt from the augmented label map on every sample
   (`#210`), and the model assembled outside `Engine` because a flow head is `2 + n_class`
   wide and the engine takes its width from `spec.data.n_class` - the derivation `#160` has
   yet to name. The script says so in its own docstring rather than leaving it to be
   inferred.

        134 cases, 5999 objects, 256x256     score   objects back
        foreground Dice                     0.9029
        flow angle, median of case medians    6.6 degrees
        AP flows                            0.5278           5424
        AP connected components             0.4598           4686   same mask, no signal
        TRUE identity, this mask            0.5941           6138   perfect vectors
        this field, TRUE mask               0.7676           5421   perfect mask

   **The flow signal is worth +0.0680** over connected components on the same predicted
   mask, which is the only honest comparison here - same model, same mask, same pictures,
   with and without the signal. The rim route was worth **+0.0640** on its own set
   (`#159`, `JOURNAL.md` §4bo), and the difference between the two, 0.0040, is **below the
   0.0074 run-to-run spread** of one configuration at a fixed seed that `§4bo` measured. So
   the structural argument for flows - that it spends no pixel on being a boundary, where a
   rim wide enough to learn hollows the object out - **does not show up in the score**. Two
   runs of this configuration here came back 0.5278 and 0.5315, a difference of 0.0037,
   which is consistent with that spread and is still two runs rather than a spread of its
   own.

   **And the oracle pair says why, which is the part worth keeping.** A perfect mask is
   worth **+0.2398** and perfect vectors **+0.0663**: the mask is the limit by a factor of
   3.6, and `0.7676` with a perfect mask is already about what connected components reaches
   on the truth (`0.7505`). A third separation target would be measured against the same
   wall, which is what `#213` now has to answer for rather than assume.

   Each row prints **the number of objects it handed back**, and that is not decoration. The
   first version of the TRUE-field oracle scored 0.2220 with **18758 objects back against
   5999** in the truth, because it handed `flows_from_instances(labels)` - zero everywhere
   off the truth's own objects - to the model's mask, and a zero field moves nothing, so
   every voxel the mask got wrong stood still and became a basin of its own. It is
   `truth_field_on` now, substituting **identity** and not the field: every voxel of the
   mask joins the nearest truth object and the field is built from that. An average
   precision alone cannot tell a route that mislabels from one that shatters; an object
   count can. One blind spot is left and it is counted rather than argued about: **259 of
   the 6138 objects** the TRUE-identity row returns are pieces of predicted foreground
   holding no truth voxel at all, which join the nearest object by identity but, under
   `diffusion`, receive no heat and keep a zero vector. Under `centroid` they would cost
   nothing - a straight line to a sink needs no path - and the run prints the count beside
   the row with the mode it belongs to.

 - **`separation_ceiling.py --separation hv`**, which is `#213`'s own cheap test and answers
   it before a model is trained. The issue, after `#211` narrowed it: *"if the markers from
   `hv` do not beat eroded interiors on the ground truth, they will not beat them predicted
   either, and this issue closes before a model is trained"*. They do not. Both recipes for
   getting markers out of HoVer-Net's two distance maps are swept, over the 134 held-out
   cases and 6000 objects:

        marker source                 AP    over the bar   markers   objects with none
        eroded interiors, 1 px    0.9807        +0.2224      6030                   0
        hv centres, |h|,|v|<=0.6  0.9382        +0.1799      5900                   5
        hv gradient, >=0.6        0.7995        +0.0412      5404                   0
        connected components      0.7583          the bar

   The maps' own zeros - `#213`'s own narrowing, *the extrema as seeds rather than the
   gradient as a surface* - are the better of the two and still **0.0425 below a one-pixel
   erosion**. HoVer-Net's published gradient route recovers **a fifth** of what that erosion
   does over the bar. Both curves turn over at 0.6 rather than at an edge of the range
   (`--hv-sweep` prints them), so neither number is a tuning artefact, and the erosion's
   thin-object fallback cannot account for the difference: it rescues **4 objects of 6000**.

   **Two rows of that table are controls.** The connected-components bar and the
   eroded-interior ceiling reproduce `0.8.0a18`'s recorded **0.7583 and 0.9807 to four
   decimals**, which is how a reader knows this measurement is on the same pictures as those
   - and the direct answer to the provenance failure that left `#159`'s ceiling without a
   set it could be reproduced on.

   The two marker functions live in the example rather than in the package, because `#160`
   has not named `separation: hv` and this measurement is the reason it may never need to.
   Seven mutations, six caught; the seventh is named in the code as uncatchable with the
   reason, rather than left as a gap.

   **`--separation both` is now `--separation all`**, because there are three signals.

### Fixed

 - **`labels` without 0 is refused when any voxel matches none of them** (`platypus#153`).
   Index 0 means two things in this package and always has: the first entry of a naming, and
   *matched nothing*. `labels: [0, ...]` makes them agree, and `labels` is the one naming
   where the 0 can be left out - after which every unmatched voxel, which is usually most of
   the image, is trained as the **first declared class**. Measured on the issue:
   `labels: [1, 2]` over a mask of 0s, 1s and 2s put the background into class `1` with an
   unmatched fraction of 0.5, and nothing in the shape of the target gives it away - it is in
   range, one-hot and plausible.

   Refused in `fit`'s existing mask check, which already reads a sample of the training
   masks, and **only when the data actually has unmatched voxels**: `labels: [1, 2]` over
   masks that really do fill every voxel is correct and keeps working. That was the author's
   reading on the issue, between four options, because the failure is in the data rather than
   in the configuration.

   The threshold is **any unmatched voxel at all**, not the 5% the unmatched *warning* uses,
   and that is a calibration rather than strictness: a label map holds integers, is read
   nearest-neighbour and compared with half a unit of tolerance, so a naming that describes
   it matches **exactly** everything. The compression and antialiasing the 5% exists for
   belong to colormaps, and a colormap cannot omit a background entry. Five mutations, five
   caught - including the one where the refusal is hung on the warning threshold, which the
   fixture is deliberately sized to catch at 1.6%.

## [0.8.0a18] - 2026-10-12

The first half of `flows` (#212): the target, its decoding, and their ceiling measured on the
ground truth **before anything is trained** - the order `#159` was done in, now the pattern
rather than a one-off. `#212` stays open; what is here is everything that can be settled
without a training run, and it settled more than expected.

### Added

 - **`flows_from_instances()` and `instances_by_flows()`, with `FLOW_MODES` as the question
   they exist to answer.** A unit vector per foreground pixel pointing at its own object's
   sink, decoded by following the field until the pixels that arrive together are one object.
   No interior, no rim, nothing eroded - which is the premise of `#212`, because `#159` closed
   measured-and-no for a structural reason: an interior and a rim are two classes competing
   for the same pixels, so a rim wide enough to learn hollows the object out. A flow field
   spends no pixel on being a boundary, and an object one pixel wide is an ordinary object
   with a sink in it rather than a case for `parts_from_instances`'s fallback.

   **The ceilings, one run over 134 held-out Data Science Bowl cases and 6000 objects:**

        flows:diffusion + follow     0.9924      6043 objects back
        flows:centroid  + follow     0.9478      6268
        rim + nearest_seed           0.9807      6074
        rim + watershed              0.9703
        connected_components         0.7583      the bar any model using it is capped by

   So the flow route's ceiling is **0.0117 above the rim route's** on the same pictures, and
   the route is not what caps a model either way. `separation_ceiling.py --separation both`
   prints all five rows and the set they belong to.

 - **`follow_flows()` exposed on its own**, because the decoding's parameters are not equally
   dangerous and only a measurement says which is which. The **step count does not matter**:
   AP 0.9914 at one step against 0.9923 converged over fifteen cases, a span of 0.0009, below
   the 0.0013 run-to-run spread of a training run. The reason is structural - the field is
   continuous, so one object's endpoints stay a connected set however far along they get, and
   it diverges at a seam, so the gap between two objects opens on the first step. **The step
   size is a cliff**: 0.9923 at 1.0 or below, **0.4715 at 2.0** with 868 objects back where
   the truth has 633, and 0.0303 at 10.0 with 4337. A step longer than a pixel jumps past the
   sink and keeps jumping. Both are in the suite with both ends written down.

### Measured, and it changed the design

 - **The two flow targets differ by 0.0446 of AP, and the obvious explanation was measured at
   0.32%.** The cheap mode aims a straight line at the sink, and the reading everybody reaches
   for is that it fails on concave objects whose centroid falls outside them - of 6000 objects
   **19 do**. The cause is one level down: a straight line between two pixels of a *discrete*
   set can step outside the set even when the set looks convex, because the boundary is
   ragged. Measured on 274 of those objects: 0.27% of foreground pixels have such a path, 6
   objects come back split, and **every one of the 6 has pixels whose path leaves it** while
   10 objects have such pixels and survive. Necessary, not sufficient.

 - **The build cost is the number that decides the training half, and it is 23x apart between
   the modes**: 4.9 ms per case for `centroid` against 113.6 ms for `diffusion`, with decoding
   at ~335 ms for both. `#210` says the target is rebuilt from the augmented label map on
   every sample, so the build is paid once per image per epoch: 429 training cases at 113.6 ms
   is **48.7 s against a 15.5 s training epoch**, 3.1x it serially and absorbed by the
   loader's four workers rather than eliminated. The decode is paid once per evaluation and
   the step-count measurement above makes it twenty times cheaper than its default.

### Fixed

 - **The diffusion target carried a full unit vector at every object's sink**, found by the
   mutation gate rather than by a test: `(0.707, 0.707)`, because normalising the residue a
   central difference leaves across a maximum produces a unit vector from nothing. A model
   trained on that learns noise at exactly the pixel the method is about. **The decoding
   survived it unchanged** - 0.9925 before the fix against 0.9924 after - which is why nothing
   noticed, and is the shape of the whole category: the defect was in the target and the only
   instrument pointed at the decode. The sink is now zeroed by construction, and the test that
   should have caught it is parametrised over both modes; with the default mode alone it was
   asserting a property of the *straight-line* field.

 - **`instances_from_parts`'s docstring said `scikit-image` is not a dependency.** It became a
   hard one in `0.8.0a16` and nothing re-read the sentence.

### Measured and unresolved

 - **The ceiling of 0.9846 and the bar of 0.7655 on the record for "the 134 held-out cases"
   are not reproducible from anything on this disk**, and `separation_ceiling.py` now prints
   the case and object count beside every number so this cannot recur. Four candidate sets
   were measured: the 134 directories of `stage1_validation` give **0.9807 and 0.7583** over
   6000 objects, the seed-1 fifth of `stage1_train` gives 0.9905 and 0.7876 over 107 cases and
   4044 objects, all 536 of `stage1_train` give 0.9818 and 0.7618 over 23461, and the same 134
   at the training resolution give 0.9627 and 0.7504. The recorded object count of **5774
   appears in none of them**.

   **No conclusion drawn against that bar changes** - `#159`'s 0.5587 and `#211`'s 0.5561 are
   below every one of these - but a new number cannot be put beside an old one, which is why
   the table above is one run over one set of pictures and quotes nothing from before it.

## [0.8.0a17] - 2026-10-12

The two gates in front of `flows` and `hv` (#212, #213), and neither is about either of them:
a continuous target has nothing in the ten overlap-and-cross-entropy losses to be trained
against (#209), and augmented as a mask it comes back geometrically consistent and physically
wrong (#210). Both were to be settled before any number produced with them could be believed.

### Added

 - **`mse`, `l1` and `smooth_l1`, and the question they exist to force: where is the target
   defined?** (#209) A continuous instance target - a flow field, or horizontal and vertical
   distances - is defined on the objects and undefined off them, so a builder writes zero on
   the background. Measured on the 536 Data Science Bowl cases: the foreground is **13.97% of
   the pixels at 256x256** (13.48% at native size, 0.03% to 59.2% per case, 167 cases under
   5%), so an unmasked mean spends 86% of the objective on a value a single output bias
   satisfies exactly.

   So `where:` is a field rather than a convention. `foreground` divides by the mask's own
   total, which makes the number mean *average error per object voxel*; `everywhere` is the
   other defensible answer and is what CellPose and HoVer-Net's plain regression terms do -
   their background zero is a real value rather than a missing one. `foreground` is the
   default because it is the reading whose number means something, and because the trap is
   otherwise on by default.

   **The mask is asked of the loader, never read off the target.** `needs_mask` and
   `loss_needs_mask(spec)` are the same mechanism `boundary` uses for its distance map, for
   the same reason - the loader is built before the loss. Reading it off the target is the one
   shape this family refuses: a flow field is exactly zero at the centre of every object and
   the `hv` maps are each zero along a line through it, so "the voxels where the target is
   non-zero" excludes precisely the seeds - and #211's oracle measured that the seeds, not the
   seams, are what limits this approach. A mask inferred that way would be wrong in the one
   place the method depends on and right everywhere a test would look.

   **Nothing activates the prediction**, which is the other sigmoid-shaped way to be wrong
   here: squashing an unbounded flow into (0, 1) would still return a number and a gradient.

 - **`onehot_to_foreground()`**, one home for a convention the package already depends on
   everywhere: channel 0 is background in the two exclusive namings, because
   `labels_to_classes` and `colours_to_classes` both send an unmatched voxel to index 0, and
   `regions` has no background channel at all. Its own docstring states the limit, which is
   filed as `maju116/platypus#153`: `labels: [1, 2]` makes the first entry a real class and
   there is then nothing that means background.

### Decided

 - **A vector target is rebuilt from the augmented label map, never corrected** (#210). Two
   measurements, both in `tests/data/test_vector_targets.py` where a library bump re-runs
   them:

   * **albumentations 2.0.8 never invents a label value.** Sixteen transforms, twenty seeded
     draws each, a label map of forty distinct ids: not one output value was absent from the
     input, and the control is that the same transform with the map sent down the *image* path
     invents hundreds. So an *instance* label map travels through the augmenter unharmed and
     the target can be built from what comes back.
   * **A crop moves an object's centre, and no correction can answer that.** Negating x after
     a horizontal flip is exact - 0.0% of foreground voxels disagree with a field rebuilt from
     the flipped labels, at 0.0 degrees, against a median of 90 and a maximum of 167.3 if the
     negation is left out. After a 32-pixel crop **49.5% disagree, by a median of 8.9 degrees
     and up to 180**, and there is no component to negate: the surviving part of an object has
     a new centre, so the arrows carried in point at a place that is no longer in the picture.

   Rebuilding is therefore correct by construction for every transform rather than for an
   admissible list of them, which also retires the idea of refusing transforms by name. It
   covers tiling for free - a tile is a crop, it happens before augmentation, and a
   2048-pixel retina cut into sixteen would otherwise carry sixteen fields built for the whole
   picture.

   The data path already had the right shape and the wrong array in it: masks are augmented as
   class indices and one-hotted afterwards, which is exactly the route, but `_to_classes`
   throws instance identity away first. So #212 carries the *instance* label map into the
   dataset rather than deriving a target offline, and a test pins that ordering instead of a
   docstring claiming it.

 - **`AlbumentationsAugmenter` refuses a floating-point mask**, which is what keeps that
   decision from being quietly undone. A flow field sent through as a mask comes back the
   right shape, the right dtype and the right range with its arrows pointing the wrong way;
   the loss falls, the model trains, and nothing downstream can tell. Every path in this
   package sends an integer - `labels_to_classes`, `colours_to_classes` and `labels_to_regions`
   all return `int64` - so the guard costs nothing it should not.

## [0.8.0a16] - 2026-10-12

### Added

 - **`instances_by_watershed()`: a second decoding of the `rim` target, and `scikit-image` as a
   hard dependency.** One trained model serves both decodings, because they share a target and
   differ only in how its channels become objects - which is why `decode` will be a *list* on
   an instance model and `separation` will not (#160). A decoding costs **11 ms per image**
   against 15.5 s per training epoch, so it is a row in a table rather than a run.

   `scipy.ndimage.watershed_ift` is not a substitute: it wants integer markers and implements
   the image-foresting-transform watershed, a different algorithm from the one this literature
   means. `scikit-image` is hard rather than an extra for the reason `scipy` is, plus one more:
   every published comparison on this problem uses `skimage.segmentation.watershed`, so writing
   our own would make a difference from the literature indistinguishable from a defect in ours.

   **What to pass as the surface is the part the name gets wrong.** Not the image - a nucleus
   is bright and so is the next one, and the seam between two touching nuclei is not reliably
   darker than either. The model's own predicted rim probability is the surface, which is the
   information `nearest_seed` throws away: it thresholds the rim and never looks at it again,
   so a confident boundary and a marginal one move the seam by the same amount.

### Measured and no

 - **The watershed does not help, and an oracle says why** (#211). Four decodings of one
   trained `rim` model at width 3, its optimum:

        nearest_seed, distance only                 0.5561
        watershed, predicted rim probability        0.5458     -0.0103,  7.8x sd
        watershed, the TRUE rim as surface          0.5590     +0.0029,  2.2x sd
        connected_components, signal discarded      0.4988

   **sd 0.0013**, from three runs of the identical configuration - 0.5587, 0.5574, 0.5561.
   (The 0.0074 quoted in the entries below is the *range of two* runs at width 1; at width 3
   the spread is a quarter of that, and §4's warning about a sample of two applies to the
   looser figure.)

   The hypothesis was that the watershed is held back by its surface, rim Dice being 0.7292.
   **Half right, and the half that is wrong is the one that matters.** Surface quality is the
   largest effect in the table - the oracle beats the predicted surface by 0.0132, ten times
   the sd. And a *perfect* surface still only draws level with ignoring the surface entirely.

   So the limit is not where the seam falls: **both decodings share the same seeds, and the
   seeds decide which objects exist at all.** Perfect seam placement buys 0.003 of the 0.21
   still missing to the 0.7655 bar.

### Changed

 - **A ceiling run cannot test the watershed, and `separation_ceiling.py` now says so beside
   the number.** With perfect parts the rim is binary, the surface is flat and the flood has
   nothing to climb, so `rim + watershed` reads 0.9756 there against nearest-seed's 0.9846 -
   an artefact of flooding order rather than a comparison. A test asserts the caveat is
   printed, because the figure without it is misleading in the most-read place.

 - **`train_instance_separation.py` reports four decodings of one forward pass**, including the
   oracle row - predicted seeds, true surface - labelled a diagnostic and not a route. That
   pattern is how the above was settled at the cost of one row and no extra training, and it is
   the pattern `#212` and `#213` should both inherit.

### Measured and no, from 0.8.0a15's work

 - **A boundary separation signal is not worth it on this data** (#159, closed
   measured-and-no). `0.8.0a15` below measured the *ceiling* of interior-plus-rim separation at
   **0.9846** and set the bar for "separation did something" at **0.7655**, which is where a
   perfect semantic mask with no separation at all already lands. Four runs then measured what
   a model actually reaches:

        rim   fg Dice  interior  rim Dice  objects  matched@0.5   AP sep   AP cc   effect
        1 px   0.9188    0.9012    0.4808     4923         4020   0.5169  0.4893  +0.0276
        2 px   0.9199    0.8908    0.6419     5086         4248   0.5486  0.4907  +0.0579
        3 px   0.9186    0.8554    0.7292     5481         4513   0.5587  0.4947  +0.0640
        4 px   0.9171    0.8068    0.7622     5525         4438   0.5306  0.4791  +0.0515

   **The best is 0.5587 and the curve turns over**, so the optimum is bracketed rather than
   guessed at. The mechanism is in the two middle columns: a rim wide enough to learn is wide
   enough to hollow the object out, so rim Dice climbs 0.48 to 0.76 while interior Dice falls
   0.90 to 0.81. Two runs of one configuration differ by 0.0074, so the best effect is 8.6x
   the noise and the turnover 3.8x - real, and small.

   The mask never moved across the four (foreground Dice 0.917-0.920 against the two-class
   model's 0.9205), which is what makes the comparison one variable rather than two.

   **This entry is here because the `0.8.0a15` notes below would otherwise read as an open
   recommendation.** They end with the bar and the ceiling and no outcome, and a changelog
   entry about work in flight goes stale the moment the flight lands.

### Added

 - `examples/train_instance_separation.py` - the script every row above came from, kept because
   a negative result nobody can reproduce is an opinion. Not packaged in the wheel, like the
   rest of `examples/`, which is why this is not a release.

## [0.8.0a15] - 2026-10-12

### Added

 - **Instance separation, and the measurement that says it is worth training** (#159). The
   proposal was a semantic mask plus a separation signal; of the two signals on the table this
   takes the boundary, because it turns the problem into **three mutually exclusive classes** -
   background, object interior, object rim - which is ordinary semantic segmentation on the
   softmax path this package already has. Nothing new in the model, the loss or the metrics.

   `parts_from_instances()` decomposes an instance label map into those parts and
   `instances_from_parts()` grows them back: the interiors become seeds and every other
   foreground pixel joins **the nearest seed**, which `distance_transform_edt(return_indices=
   True)` answers in one call. No new dependency - `scikit-image` is not here and is not worth
   adding for one function.

   **No model is trained, and that is the point of this release.** The post-process is what
   caps the approach, and a cap can be measured without the thing it caps: feed the ground
   truth through it and score the result against the same ground truth. On the exact 134 cases
   `dsbowl-unet`'s card reports:

        interior and rim, grown back        0.9846    median 1.0000
        connected components                0.7655
        the published weights, as measured   0.4888    (0.8.0a13)

   Three readings, and the middle one is new. **Connected components caps any model at
   0.7655** on this data - a perfect semantic mask still scores that, because touching nuclei
   are one region. So the published weights' 0.4888 is not one failure but two: 0.4888 to
   0.7655 is the mask's own error, and 0.7655 to 0.9846 is what separation can buy and no
   better mask ever will. `examples/separation_ceiling.py` is the script, it trains nothing,
   and it prints both ceilings from one run over one set of pictures.

   What the decomposition costs, counted rather than assumed, over those 134 cases: the plain
   erosion would have **erased 6 objects in 4 cases** - a nucleus one or two pixels wide has no
   interior - and an interior falling into two pieces **gained 52 objects in 20 cases**, which
   is most of the 0.0154 the ceiling sits below 1.0.

   The erased objects are handled rather than ignored, because an object absent from the target
   is an object the model is told not to find: a false negative in every epoch and in the
   score, with nothing to say the target was at fault. An object whose interior comes back
   empty falls back to an erosion that only removes pixels touching a *different* object, and
   one still empty after that keeps itself whole.

### Fixed

 - **The weak erosion could not see the objects it was protecting against.** It was handed a
   label map with every other object masked out, so a pixel touching a neighbour read
   background there - and background counts as agreement in that pass by design. The thin
   object was kept whole and handed a seed it shared with its neighbour, which is the one
   outcome that fallback exists to prevent. Found by a test, not by reading, and the ceiling
   above was re-measured afterwards: it had been 0.9853 with the defect, which is wrong in the
   third decimal and was nearly published.

### Notes

 - Nothing in `pyplatypus.instances` is in `pyplatypus.__all__`, deliberately: the top-level
   namespace is paired with R's reference groups and the shared order is held by tests on both
   sides. These names live on `pyplatypus.instances` until R has something to say about them.

### For the R package

 - **Nothing, and the pin should not move for this.** No R surface reaches the instance
   post-process yet; `platypus#103` is the R half of the whole milestone and it waits on #160,
   the spec tag, which is what will make any of this reachable from a configuration.

## [0.8.0a14] - 2026-10-12

### Added

 - **`regions`: overlapping classes, named as unions of a label map's values.** The first half
   of multilabel segmentation (#91), and the decision in front of it was the author's: nested
   regions first, one binary file per class deferred.

   The case is BraTS and the lesion-inside-an-organ family - liver and tumour, kidney and cyst,
   lung and nodule. The ground truth gives every voxel one of `{0, 1, 2, 4}` and the three
   scored classes are unions of those values:

        regions:
          - name: enhancing
            of: [4]
          - name: core
            of: [1, 4]
          - name: whole
            of: [1, 2, 4]

   So **the overlap is a convention applied to exclusive data**, not a different file format,
   and nothing new is read off disk. `regions` is a third way of naming classes rather than an
   addition to `labels`: a region is a membership test, so the complete vocabulary of values
   never has to be declared. Exactly one of `colormap`, `labels` and `regions` is given, and -
   as with detection's `classes`, for the same reason - **regions have no background entry**,
   because each channel is an independent binary question.

   **Why nested first was the cheaper half as well as the one the data wanted.** One integer
   per voxel survives tiling and augmentation, both of which operate on a single-channel array
   today, so neither had to learn anything. The expansion to channels stays in the last step
   before the loss, where `classes_to_onehot` already was:

        exclusive    onehot = np.eye(n_class)[classes]
        overlapping  onehot = table[classes]      table[v] = the regions value v belongs to

   Same indexing; `np.eye` is the special case where the table is the identity. The one
   difference that matters is **row 0, which is zeros rather than a class**. `labels_to_classes`
   sends an unmatched voxel to index 0 because index 0 *is* a class there - the background a
   colormap puts first - so reusing it unchanged would have made background a member of
   whichever region holds the lowest value. On BraTS that is 1, which is in both `core` and
   `whole`, so every background voxel would have been in two of the three regions and the
   model would have trained on it without complaint.

 - **Per-channel objectives, and the four places mutual exclusivity was hiding.** `probabilities`
   gained a `multilabel` argument and is the single home of the softmax-or-sigmoid decision;
   every overlap-based loss reads it through `SegmentationLoss.probabilities` rather than
   calling the functional form, so Dice, IoU, Tversky, focal Tversky, combo and the surface
   term all moved with one line. `cce` and `focal` gained binary forms, metrics threshold each
   channel instead of taking an `argmax`, and `predict` returns a stack whose channels do not
   sum to one.

   **`binary_focal` agrees with `torchvision.ops.sigmoid_focal_loss` to within float32's
   resolution** over five (gamma, alpha) settings, and reduces *exactly* to
   `binary_cross_entropy` at `gamma=0` - a control the implementation could not satisfy by
   accident. Both are in the suite, and the torchvision numbers are baked in rather than
   imported because torchvision arrives only with the `encoders` extra.

   No tighter number than that is pinned, because there is not one to pin: these are float32
   reductions over 384 elements, so the last digit belongs to the order the hardware adds them
   in. macOS on ARM and Linux on x86 disagree by **8.6e-08 relative**, under float32's eps of
   1.19e-07, and a first version of the test asserting 2e-08 absolute turned one macOS job red
   while being right about the arithmetic. The bound that replaced it is `rel=1e-6`, chosen
   against both ends: twelve times the platform noise, and five orders of magnitude under the
   55% to 67% that the mutation this test exists to catch actually moves the numbers by.

   **The reduction differs from the categorical form and that is deliberate.** Categorical
   cross-entropy sums over channels, which picks out the one true class of a one-hot target.
   Here every channel contributes, so a sum would make the loss grow with the number of
   regions: three regions would read three times the scale of one, and `cce_weight` in a
   blended loss would mean something different on every dataset. The mean keeps one voxel worth
   the same whatever it was asked. `label_smoothing` divides by two rather than by `n_class`
   for the same reason - each channel is a two-outcome question.

 - **A metric now reports what the model was asked.** With an `argmax`, a voxel correctly
   predicted in both the core and the whole tumour was credited to whichever channel read
   higher and scored as a miss in the other. On the smallest case that can show it - one voxel,
   in two regions, both predicted - Dice went from 0.833 to 1.000.

### Changed

 - **`inspect_masks` reports regions by name, and `unmatched` means something different there.**
   `missing_classes` keeps its meaning exactly and gains the failure `regions` invites: `of: [3]`
   against data holding `4` is a channel of zeros, which trains to predict nothing and scores a
   Dice of 0 while every other number looks ordinary. It is refused before training with the
   region named. `unmatched`, on the other hand, counts background - which is deliberately not
   a region - so it sits near 1 on any dataset where the thing being found is small. On BraTS
   that is the correct answer and it looks like the catastrophic one, so the warning that reads
   it is suppressed for `regions` rather than left to cry wolf on every correct run.

 - **The class fingerprint written into every run record and weights sidecar now names the
   regions.** `n_class` alone cannot tell three exclusive classes from three overlapping ones,
   and weights from one must not be read back as the other.

### Refused rather than served

 - **`lovasz` with `regions`.** Lovasz-Softmax sorts the errors of a softmax over mutually
   exclusive classes; the independent form is the Lovasz-Hinge, a different construction rather
   than a setting on this one. Run as it stands on sigmoid channels it would return a number and
   a gradient, which is the worst available outcome. The property lives on the loss
   *specification* (`supports_multilabel`), so the refusal arrives at load time and
   `pyplatypus.objectives.losses` reads it rather than keeping a second list.

 - **`include_background: false` with `regions`**, but only when it was written down. There is
   no background entry, so channel 0 is the first region somebody asked for and dropping it
   would drop a class. `cldice` defaults the flag to False and a default is not a request, so
   with regions the setting is ignored rather than honoured - `model_fields_set` tells those
   apart, the same way `channels_match_the_data` already does for `channels`.

 - **A colour map holding an overlap.** `onehot_to_colours` refuses an array where some voxel
   is confidently in two classes instead of taking the `argmax` and writing one of them, which
   would have produced a plausible picture of something else. Detected from the array rather
   than from a flag, so a caller holding only a prediction can answer it; a softmax output never
   trips it, because its channels sum to one.

### Not done, so it is not looked for

 - **`classes_from` - one binary file per class** - is the only shape that can express
   genuinely independent overlap, a vessel crossing an organ or two annotators' masks.
   Deferred, not dropped: #91 stays open for it and `unite_masks`'s element-wise maximum is
   still where per-file identity is destroyed.
 - **Drawing a multilabel prediction.** What one picture of three overlapping regions should
   look like is a real question with no answer yet, and half of one would be worse than none.
   `onehot_to_colours` refuses rather than guessing, which is the honest state until it is
   designed.
 - **A guide section.** The guide's section list is pinned and paired with the R package's
   ("change both or neither"), so a narrative section on `regions` lands with the R half rather
   than documenting in R something R cannot do. The field's own description is on the generated
   configuration reference now, which is where a reader meets it.

### For the R package

 - **Nothing moves yet, and the pin should not.** `regions` is reachable from R the moment the
   pin moves, because R forwards the data block it is given - but `available_weights()`-style
   free inheritance is not what this is: the R surface documents its fields by hand, so
   `regions` needs roxygen prose of its own, and `save_masks()` needs the refusal the author
   chose for an overlapping prediction. Both are the R half's work, filed as its own issue.
 - **The field-prose ledger will fire**, and that is the mechanism working: `regions` and
   `Region.of` are new descriptions R has never been written against.

## [0.8.0a13] - 2026-10-12

### Added

 - **`pyplatypus.instances`: the Data Science Bowl's own metric, and the number the published
   weights score under it.** First piece of instance segmentation, and it deliberately builds
   no model. Detection worked here because it was built metrics-first - every piece checked
   against something outside itself - and a model chosen before there is a way to score it is
   a model chosen on nothing.

   **The baseline, measured on the exact 134 cases `dsbowl-unet`'s card reports:**

        Dice, on the model's grid                   0.9205
        instance average precision, source grid     0.4888

   The card for those weights says *semantic, not instance*, and that caveat has been true and
   unmeasured since they were published. Touching nuclei come back as one region; connected
   components found 4658 objects where the data holds 5774, and every nucleus a merged region
   swallows costs twice - a false negative, and a false positive through the region that
   swallowed it.

   **The split is reproduced rather than assumed, and the Dice is the control that proves it.**
   The 134 directories sitting in `stage1_validation` on the machine this was measured on are
   a *different* 134 from the card's - the card's worst case is not among them - so scoring
   those and quoting the published Dice beside it would have compared two different sets of
   images. `split_dataset` with the card's fractions and seed rebuilds the right ones, and the
   script refuses to report anything unless Dice comes back as the sidecar's
   `0.9205236150257623` to seven decimals. That mistake has a history on this dataset: a
   directory of 134 images and a card measuring 134 held-out images were taken for the same
   134 once before.

   What is in the module:

   * `label_map()` and `read_instances()` - the per-object files read as instances instead of
     united into one mask. `unite_masks()` takes the element-wise maximum, which is where
     instance identity is thrown away at read time; nothing new had to be collected.
   * `iou_matrix()` - pairwise IoU from a contingency table rather than pair by pair, because
     one image in this dataset holds 375 objects and the pairwise form is 140,000 full-image
     comparisons. **Checked against `pycocotools` to 1e-12** on three seeded problems, then
     baked in, so the suite needs no dependency - the arrangement the detection metrics
     already have with that library.
   * `average_precision()` - the challenge's `TP / (TP + FP + FN)` at ten thresholds from 0.5
     to 0.95, averaged. Not a precision-recall curve and nothing interpolated, because nothing
     is ranked: an instance post-process produces regions, not scored proposals.
   * `instances_from_mask()` - connected components, 4-connectivity, untuned. A yardstick, not
     a contribution, and a baseline that has been tuned is no longer what later work is
     compared against.
   * `summarise_instances()` - the mean over images, which is what the challenge ranked on,
     **and** the pooled number beside it. This dataset holds images of 2 objects and images of
     375, so the two differ and reporting one alone is how it gets quoted as the other.

   **Thresholds below 0.5 are refused, and the strictness of the comparison is load-bearing.**
   Above 0.5 a prediction can satisfy at most one object - two disjoint regions cannot each
   cover more than half of the same third region - so counting matches is exact and needs no
   assignment algorithm. The comparison is `iou > threshold`, not `>=`: with `>=`, two
   predictions each covering exactly half of one object would both count, and the test that
   catches this is the one where a merged region scores two true positives instead of none.

   Overlapping truth is refused rather than resolved. A mask layer that holds more than one
   thing per pixel is open work - `pyplatypus#158` with `pyplatypus#91` - and a metric that
   quietly took the topmost label would answer that question wrongly rather than decline it.

   Closes pyplatypus#157.

### For the R package

Nothing, and the pin stays where it is. These names live under `pyplatypus.instances` rather
than in `__all__`, so they are not on the reference page and R cannot reach them: instance
segmentation becomes a configuration people write when `task: instance_segmentation` lands
(`pyplatypus#160`), and that is the release where the reference groups and R's shared order
move together. `platypus#103` is the R issue and it says the same thing.

## [0.8.0a12] - 2026-10-12

### Added

 - **`fives-unet`, the third published weights, and the first that are not MIT.** A U-Net on
   512x512 tiles over FIVES' 2048x2048 fundus photographs, trained on the dataset's own split
   and pinned in the registry at `c4492f7a`. `weights: fives-unet` fetches it, the sidecar
   refuses a model it does not belong to, and nothing about the numbers can change under the
   name.

   Measured on the 200 held-out images, seed 1, one score per image on masks reassembled from
   the tiles:

        Dice      mean 0.8789  sd 0.1051  median 0.9092  worst 0.1055 (122_G)
        IoU       mean 0.7956  sd 0.1280  median 0.8335
        clDice    mean 0.8330  sd 0.1268  median 0.8714

   40 epochs were asked for; `early_stopping` stopped at 24 and restored epoch 16. The
   previous 40-epoch retinal run measured 0.8509, so stopping earlier cost nothing.

   **The per-disease table reads picture quality as pathology, and the card says so before it
   says anything else.** FIVES ships a quality grading - every image rated for illumination and
   contrast, blur, and low contrast - that nothing in the training pipeline uses, and **14 of
   the 15 test images flagged for low contrast are glaucoma**:

                              all images   graded clean on all three axes
        AMD                   0.9107       0.9229  n=42
        diabetic retinopathy  0.8818       0.9056  n=36
        glaucoma              0.8347       0.9315  n=22
        normal                0.8884       0.9006  n=33

        graded clean   n=133  dice 0.9141
        graded poor    n= 67  dice 0.8089

   Glaucoma is the weakest group over all 200 images and the **strongest** over the 133 graded
   clean. The 0.1052 between clean and flagged images is larger than the 0.0761 between the
   best and worst disease, and larger again than the 0.0309 between them once the flagged
   images are set aside. A card reporting `glaucoma 0.83` beside `normal 0.89` would invite a
   clinical conclusion that is wrong, so every figure on it is generated from the run rather
   than typed, and the generator refuses to print if that ordering ever stops holding.

   **`weights_licence` is CC BY 4.0, not MIT.** `dsbowl-unet` was trained on CC0 data and
   `bccd-yolo3` on MIT, so neither constrained anything. FIVES is CC BY 4.0, these weights are
   plausibly a derivative work, and attribution therefore travels with them: the sidecar
   carries the citation, the source and the licence, the registry description says so, and the
   Hub repository now declares `license: other` with `license_name: mit-and-cc-by-4.0` because
   one field cannot describe three files under two licences.

 - **`examples/publish_fives_weights.py`**, which produced them. The configuration is imported
   from `examples/segment_retinal_vessels.py` rather than copied, so the published weights are
   the run that example documents and the two cannot drift. It uploads nothing, on purpose. Its
   one deviation is a `model_checkpoint` watching `val_dice` - the metric the example's
   `early_stopping` watches, because `restore_best` hands back the epoch the stopping rule
   chose and two monitors would leave a file on disk from one epoch while the published weights
   came from another.

   It also refuses to write anything if the mean Dice comes in under `--floor 0.80`, or if the
   quality grading does not hold the three counts every claim on the card depends on: 200
   graded, 15 flagged for low contrast, 14 of those glaucoma. Under the opposite polarity it
   would be 185 images - not a flag - and the glaucoma share would be chance.

 - **The published file was loaded the way a user reaches it, before this entry was written.**
   `weights: fives-unet` with `fit: false`, resolved through the registry, downloaded from the
   pinned commit into a cache that had never seen it, sidecar checked, and scored on the same
   200 images: Dice, IoU, clDice and all four graded-clean per-disease means came back
   **identical to the published sidecar, to every digit**, and the downloaded file is
   byte-identical to the local one. Green on both halves proves the halves; this is the join.

### For the R package

The pin moves to `0.8.0a12` and nothing else is needed: `available_weights()` reads the
registry from the engine, so the third name appears on the R side as soon as the pin does.
`platypus#131` is the issue, and it stays open until that lands.

## [0.8.0a11] - 2026-10-11

### Added

 - **`deterministic`, the opt-in that makes a seed mean something on a card.** `seed` has
   said where it stops since `0.8.0a8`: measured on a GTX 1070, two runs of one
   configuration at one seed agreed to twelve decimals on the processor and differed in the
   fourth after a single epoch on CUDA, because cuDNN picks algorithms nothing constrained.
   This constrains them - `torch.use_deterministic_algorithms(True)`,
   `cudnn.deterministic`, and `cudnn.benchmark` off, which is the one that was choosing per
   shape.

   **An opt-in rather than the default, because the cost arrives at a time nothing else here
   does.** Some convolutions become much slower, and an operation with no deterministic
   implementation raises *when it is reached*, so a run can stop ten minutes in. Every other
   refusal in this package happens before anything is read, and this one cannot: whether an
   operation has a deterministic algorithm is a fact about the installed torch and the
   device, not about the configuration. The field says so.

   **It needs `seed`, and asking without one is refused.** Constraining the algorithms while
   the initialisation, the shuffle and the augmentation all still vary leaves the run exactly
   as unrepeatable as before and only slower - half a promise, which is what platypus#122 was
   filed about in the first place.

   It also sets `CUBLAS_WORKSPACE_CONFIG` to `:4096:8` when nothing else has: from CUDA 10.2
   on, cuBLAS needs it or torch raises, and the variable is read when CUDA initialises - which
   is why this happens in the engine's constructor and not in `fit()`. An existing value is
   somebody's deliberate choice and is left alone.

   What is tested is that the flag reaches torch, that **nothing is asked of torch when
   nobody asked**, that the variable is set without trampling one, that the specification
   refuses the half-promise, and that an ordinary run still trains with it on. What the field
   is for - two CUDA runs agreeing - needs a card, and CI has none; `tests/test_seeding.py`
   says the same about the other half rather than asserting something it cannot see.

   Closes platypus#122, whose documentation half shipped in `0.8.0a8`.

 - **`save_spec()`, the way back out of a specification.** `from_yaml()` read a
   configuration and nothing wrote one, so the two interfaces were one pipeline and a half:
   an example could write the YAML of the run it had just done only by keeping the raw dict
   it started from. Now a specification built in code - or in R, where the bridge already
   drops unset arguments - becomes a file somebody else can run.

   **It writes what was set, not what the object holds**, and that is the whole design. On a
   two-field model `to_dict()` returns 29 fields, and the difference is not verbosity: a
   model given `weights` and no `architecture` adopts the architecture from the weights
   sidecar, and the adoption fires on `architecture` *not being in* `model_fields_set`. Dump
   the defaults and the reloaded specification has all 29 set, so a file that looks like the
   original quietly stops doing what the original did. `defaults=True` is the opposite want,
   kept as a second mode: freeze this run exactly as it resolved, at the cost of it adopting
   nothing ever again.

   Two things only the round trip could have found, both now tests:

   **Every discriminator has to be written even when unset.** `task` is the obvious one.
   `DiceLoss(smooth=0.5)` is the one that is easy to miss: `name` is a `Literal` with a
   default, so a dump of what was set gives `loss: {smooth: 0.5}` - a block with no tag,
   which comes back as *Unable to extract tag using discriminator 'name'*.

   **And "annotated `Literal`" is the wrong rule for finding those tags.** It also catches
   `box_loss` and `annotation_format`, which are choices a user makes, and writing them
   unset would freeze exactly the kind of default this function exists to leave out. A
   discriminator pins one value, so the rule counts them: measured across the spec, 31
   single-value `Literal` fields, every one of them `name` or `task`, and the only two with
   more are those choices.

   Closes platypus#120. `from_yaml`'s docstring promised this round trip through `as_dict()`, a
   method no spec has ever had; it now names the function that exists.

 - **`available_losses()`, `available_metrics()`, `available_models()`,
   `available_optimizers()`, `available_callbacks()` and `available_tasks()`** - the rest of
   the verb `available_transforms()` and `available_weights()` already answered. The number
   of named components keeps growing, and the only way to see what is on offer was to leave
   for a documentation site.

   Three columns, in the order a reader needs them: the name you pass, the arguments you may
   tune, and what it is for. Every row is read from the same pydantic models the schema is
   generated from, so **a loss added to the engine appears in the listing with nothing else
   edited** - and the tests compare each listing against the schema's own
   `discriminator.mapping`, which is the authoritative answer to *what may I write here*,
   rather than against a list of names in a test file.

   The description shown is the component's first paragraph, which is the summary line; the
   rest, including every measurement, stays in `spec_schema()` and in `help()`.

   **Detection is the case that does not fit, and it refuses rather than inventing rows.** A
   detection spec chooses no loss component - the objective is fixed and `box_loss` picks the
   box term - and chooses no metrics at all, since `map_50`, `map_50_95` and
   `mean_matched_iou` come out of `report`. Both raise with a message naming what detection
   actually offers: the `box_loss` values are read from the schema, and a test ties the three
   metric names to the keys of the report itself, so a rename fails here rather than in
   somebody's console. An empty table would have been a third answer and the wrong one.

   The architectures are the one hand-written part: they are an enum, and `UShapedNet` is a
   single class taking the choice as a switch, so there is no docstring per member to read.
   The four notes say what the code does differently - nested skips and the only architecture
   that can take `deep_supervision`, residual blocks, additive skips and the memory they save
   - and a test requires every member of the enum to have one.

   Closes platypus#119, which was blocked on the descriptions below.

 - **26 of the 29 things you can name in a configuration now say what they are.** Three did:
   `boundary`, `cosine_annealing` and `swa`, the three written most recently with a
   measurement in hand. Every one of the 201 described *fields* had its text, so `smooth`
   explained itself while `lovasz` did not say what a Lovasz loss is for - which left the
   generated configuration reference printing a heading and a table of knobs with nothing
   under the heading, and left `?losses` in R listing ten names without the one sentence a
   reader needs: *which of these do I want*.

   Where this project has measured something, the description carries it. `cldice` now
   states what it was worth on a real model - FIVES, 200 cases, correlation 0.9779 with
   Dice, mean difference +0.0041, 127 cases higher and 73 lower - alongside the constructed
   pair where Dice reads 0.8750 and 0.9474 and so prefers the severed vessel while clDice
   reads 1.0000 and 0.9425 and does not. Both of those pairs are pinned by a test.

   Where it has measured nothing, the description says what the thing answers and what it
   costs, and says so once rather than eight times: **no run here has compared the eight
   optimisers on segmentation**, so `adam`'s text is where that is admitted.

   Four of the descriptions state an identity, and each is now held by a test rather than by
   a claim:

       tversky(alpha=0.5, smooth=1)  ==  dice(smooth=2)        exactly, not dice(smooth=1)
       iou                           ==  dice / (2 - dice)     without smoothing, so same ranking
       focal(gamma=0)                ==  cross_entropy
       focal_tversky(gamma=1)        ==  tversky

   That last group is the useful kind of sentence for a knob - *at this value this is another
   loss* - and the kind most likely to go quietly wrong in a refactor, because every other
   test in the suite would still pass.

   A coverage test holds the level, driven from `spec_schema()` for the reason the field-level
   one gives at length: a definition is a component when its `name` is a literal, which finds
   29 and correctly excludes `AugmentationStep` and `SegmentationModel`, whose `name` is prose
   the user writes. It also refuses a description under 150 characters and one that only
   restates the component's own name, both proven able to fail.

   Closes platypus#118. Note that the configuration reference prints a definition's **first paragraph**
   by design, so the measurements live in the schema, in `help()` and - once platypus#119 lands - in
   `available_losses()` and its family.

## [0.8.0a10] - 2026-10-11

### Fixed

 - **Four transforms were reported as unable to take a volume when what they could not take
   was the probe.** `available_transforms(rank=3)` finds its answer by offering each transform
   a probe volume, and the probe was 8x8. A refusal is now retried once at 1024x1024 before it
   is believed, which is what `Crop` needs - its default crop box is 1024x1024, so no plausible
   small probe would ever have admitted it. The listing goes from 87 names to 91 on
   albumentations 2.0.8, and the four are `Crop`, `FrequencyMasking`, `TimeMasking` and
   `Superpixels`.

   **Why two sizes rather than one bigger one**, measured on one Linux machine, as a ratio
   because seconds are not portable: offering every transform the large probe takes **250x**
   the 8x8 listing, while retrying only the refusals takes **8x**. Most of that 8x is
   `Superpixels`, whose work scales with the pixel count. It is an `lru_cache`d call made once
   per process, and a spec with no augmentation never makes it.

   The escalation can only add names - a transform that passed at 8x8 is never asked again -
   and a test takes the large size away and asserts `Crop` goes back to absent, because the
   straightforward test would pass for any reason at all, including an albumentations release
   that changed those defaults.

 - **The 87-against-88 disagreement was never about the platform, and now has a mechanism.**
   It had been recorded as "Windows with Python 3.13, and no mechanism is claimed". The first
   version of the escalation test asserted that all four names were what escalation added, and
   `ubuntu-latest` with Python 3.10 reported three - which is how the real answer arrived:

       Superpixels at 8x8      4 of 60 attempts succeed; it samples `n_segments`, and
                               whether the segmentation divides by zero depends on the draw
       eight listings, 8x8     87, 87, 87, 87, 87, 87, 88, 87 - one process, one machine
       Superpixels at 1024     40 of 40 succeed
       eight listings, now     91, eight times

   So the probe was a random variable and the matrix was reading its noise. The count is still
   not pinned anywhere, because stable on one machine is not portable - but it is no longer
   unexplained, and the test now asserts only what holds on every run: escalating never
   removes a name, never adds one outside the four, and always adds `Crop`.

   **The three-channel nine are still refused, deliberately.** That is a different decision,
   not an unfinished one: a three-channel probe would list `ChromaticAberration` and its eight
   relatives as available to someone whose CT has one channel, and a listing cannot know the
   channel count. Only the pipeline-build check can, and it already probes at the model's own
   input size.

### Changed

 - **The docstring's account of the refusals was wrong, and wrong in a way arithmetic shows.**
   It said *"eleven of the 31 are the probe's limits"* while naming nine three-channel ones and
   four size-dependent ones, which is thirteen. Counting the categories instead of recalling
   them gives a different answer again - on albumentations 2.0.8, of 27 refusals:

        10   no volume support, which is the question being asked - `KeyError: 'images'`
         9   require three channels, where the probe has one
         8   require a target the probe does not supply - seven want metadata of their own,
             and `BBoxSafeRandomCrop` wants bounding boxes

   So **17 of 27 are the probe's limits and only 10 are a real no**, and the third category was
   not named anywhere before this. A new test asserts that every refusal falls into one of the
   three, which holds the *categories* rather than the counts: an albumentations upgrade is
   expected to move the numbers and is not expected to invent a new kind of refusal. If it
   does, the test names the transform and prints its message.

 - **The README no longer says how many transforms take a volume.** It said 87. That number is
   the outcome of probing the installed library and one CI matrix of twelve once answered 88
   where the other eleven answered 87, so by this project's own rule it must not be pinned at
   all, only explained - and a test now asserts the README states no such count. The file's
   README test cited *"a transform count of 97 where the measurement is 87"* as one of the four
   stale claims it was written for, and then asserted nothing about it.

 - **`CITATION.cff`'s `date-released` is the release date again.** It said 2026-10-08 through
   `0.8.0a9`, which shipped on the 10th. `tests/test_version.py` holds three places to the
   version and nothing holds this field to anything.

### Noted, not fixed

 - **Eight transforms are listed at rank 2 that no specification can name**, which is the third
   category above seen from the other side: `FDA`, `HistogramMatching`, `Mosaic`,
   `OverlayElements`, `PixelDistributionAdaptation`, `RandomCropNearBBox`, `MaskDropout` and
   `BBoxSafeRandomCrop` need a target supplied at call time, and an `AugmentationStep` is a
   flat name and a dict of parameters. The same argument that removed the ten composition
   classes in `0.7.0a2` applies to them, and it is a rank-2 change with its own blast radius,
   so it is not smuggled in here.

## [0.8.0a9] - 2026-10-10

### Added

 - **An "About the author" section in the README**, which is also this package's page on PyPI.
   It carries a photograph, the fact that both halves of the project are by one person, and
   links out. The same section, in the same place, went into `platypus`'s README - the paired
   tests on both sides hold the section order, so adding it to one would have failed the other.

   It reaches different numbers of readers on the two sides, which was not noticed until the
   release was confirmed. `platypus`'s site renders its README as the home page, so the
   section is there; this site's home page is a hand-written `index.qmd`, so it is not. The
   asymmetry is left standing rather than papered over, and nothing tests it: §3F's About
   pair holds the navbar's order, not what the home page is built from.

   **The R half of this release is deliberately not done**, by §3E: `platypus` gets nothing
   from a section on the engine's own page, so its pin stays at `0.8.0a8`. Its README got the
   same section in its own change.

   The photograph is 200 x 200 and 69 KiB. Larger than it needs to be for a 130-pixel avatar
   and smaller than two of the three figures already there, which is the trade: quantising a
   portrait to 128 colours would have saved 55 KiB and risked banding on a face.

## [0.8.0a8] - 2026-10-10

### Changed

 - **`seed` said "set it if you want a reproducible run", and that was the whole
   description.** It is now the one field on the specification that says where its guarantee
   stops, because the nuance lived in a private docstring and *half of that docstring was
   false*.

   What the field now states, measured rather than reasoned:

       repeatable      the split, a detector's fitted anchors, the order the data arrives
                       in and its augmentation - workers included - and, on the processor,
                       every number to twelve decimal places
       not repeatable  a run on a CUDA device: two runs at one seed agreed to twelve
                       decimals on a processor and differed in the fourth after one epoch
                       on a GTX 1070, because cuDNN picks algorithms nothing here
                       constrains. Nor across versions - `0.8.0a3` changed how a tiled
                       dataset is shuffled, so one seed draws a different order than it did

   The practical sentence, which is the point of the whole entry: it is the field for
   comparing two of your own runs, not for publishing a number somebody else can land on.

 - **`seed_everything`'s docstring claimed "two runs at one seed on one machine agree".**
   True on the processor, false on a card, and the card is where anybody with a GPU trains.
   Corrected, with the measurement and a note saying which half is tested and which is not.

   It also explains why the CUDA half is tested *nowhere*: a test asserting that two runs
   **differ** passes by luck and fails by luck, and CI has no card to run it on. So the
   documentation carries that half and nothing else can - said out loud rather than left as
   a gap somebody later mistakes for an oversight.

 - **`tests/test_seeding.py` names its scope.** The tests always forced `device="cpu"` and
   never said so, which made `test_the_same_seed_gives_the_same_run` read as a universal
   claim while guarding a device-specific one. Renamed to `..._on_the_processor`, with the
   module docstring pointing at `test_the_same_seed_gives_the_same_split` for the guarantee
   that holds on any device - and is the one that matters most when two results are compared.

### Note

 - **The R half is platypus#122's**, and unlike the last few §3E releases it is not nothing:
   R's `@param seed` carries the same bare sentence in its own prose, and the configuration
   reference on R's site is generated from this schema, so it shows the new text only once
   the pin reaches this release.

   Also still open there, and deliberately not decided here: whether to offer
   `deterministic = TRUE` as an opt-in. `torch.use_deterministic_algorithms` would make a
   card agree with itself at a cost this docstring has always named - some convolutions much
   slower, some raising outright. That adds a public field on both surfaces, so it is a
   decision rather than a documentation fix.

## [0.8.0a7] - 2026-10-09

### Fixed

 - **`evaluate_cases(group_by=...)` refused a whole-mask metric instead of averaging it.**
   0.8.0a6 taught `score_cases` to reassemble a case's *tiles* before measuring `cldice`, and
   stopped there. With `group_by`, several whole **images** arrive under one name - fifty
   fundus photographs under one disease, a patient's slices under one patient - and the guard
   written for the tile case fired on them:

       'cldice' reads the shape of a whole mask and this case arrived in 50 pieces

   It also named the wrong cause, suggesting several files under one key when the cause was
   the grouping. A group's figure is now **the mean of its images' scores**.

   That is a different kind of pooling from Dice's and the difference is documented rather
   than smoothed over: Dice is a ratio of sums, so a group adds the overlap counts and applies
   the formula once. A skeleton has no counts to add, so there is nothing to pool and the mean
   is the only thing the number could be. Weaker, and reported as such.

   **Found by writing the R vignette**, which is the only way it could have been: nothing in
   either suite grouped a tiled run by anything, and the combination needs a dataset whose
   case names carry a group. §4's *a feature is not shipped until something has used it the
   way a user reaches it*, on a release that was green on both halves.

### Note

 - **The R half is deliberately not done** (§3E). Nothing R sends changes; the pin moves with
   platypus#123, which is the work that found this.

## [0.8.0a6] - 2026-10-09

### Changed

 - **A tiled run can ask for `cldice` now, and `evaluate_cases` reassembles before measuring
   it.** The combination used to be refused outright. The refusal was right about the
   arithmetic - a skeleton is a property of a whole mask, and a tile severs every structure
   crossing its edge - and wrong about the remedy, which it left to the caller: the FIVES
   example carried forty lines stepping the model over one sample's tiles, stitching, and
   scoring, to get a number the engine could have produced.

   `score_cases` now holds a case's tiles until its image is complete, stitches the
   prediction and the target, and measures the whole-mask metrics once on the pair. One
   image at a time - the loader is never shuffled there, so a case's tiles arrive together
   and the buffer empties as the last one does.

   **It is still absent from the epoch**, and that is the honest half of the change. An epoch
   reads its metrics per batch and averages, which is exact for Dice - a ratio of sums - and
   meaningless for a skeleton. So on a tiled run `val_cldice` does not exist, the history
   does not carry it, and `monitorable` does not offer it, which makes the **existing**
   monitor check refuse `monitor: val_cldice` when the specification is read, with a message
   listing what is available. One validator fewer than before, not one more: the blanket
   refusal is gone and the narrow one was already there.

   The split-level figure comes from `summarise_cases` over the per-case rows - the mean over
   images, which is what a whole-mask metric's split-level value means anyway, and more
   defensible than `evaluate`'s average over batches.

 - **`examples/segment_retinal_vessels.py` lost its second pass.** `cldice` sits in `metrics`
   beside `dice`, `evaluate_cases` returns both per case, and forty lines went away along with
   a `torch` import. This is the change proving the feature: the example existed in its old
   shape only because the engine could not do this.

   Run at full scale to check it rather than assumed - 600 training retinas, 200 scored, one
   epoch, 2048 cut 4x4:

       peak memory      1.42 GB      against 2.22 for the two-pass version it replaces,
                                     and the ~20 GB that killed a finished run (#178)
       scoring          1.8 min      200 reassembled images, inside the same walk that
                                     produces dice and iou, instead of a second pass

   Less memory than the workaround, not merely less code: the workaround stitched and scored
   one image at a time too, but it did it *after* a full `evaluate_cases` walk rather than
   during one.

 - **A measured number in the configuration reference was the wrong one.** `iterations` said
   five peels is "well past a retinal vessel at one to five". A FIVES vessel is about **nine**
   pixels wide, measured as mask area over skeleton length; one to five is what a naive resize
   leaves behind. Five is therefore enough for vessels and not much more than enough, which is
   a different thing to tell a reader. The field also now says where `cldice` is reported on a
   tiled run and where it is not - on both sites, since the page is generated from the schema.

### Added

 - **`Trainer._stitch_channels_first`**, the device-side counterpart of `data.images.stitch`,
   with a test holding the two to each other over a non-square grid and non-square tiles.
   Scoring would otherwise copy a four-megapixel mask to the host and back twice an image.

### Note

 - **The R half is deliberately not done and the pin stays at `0.8.0a2`** (§3E), but this is
   the release that makes platypus#123 possible: R already has `metric_cldice()` and
   `evaluate_cases(group_by=)`, so the vignette needs no new R surface for this - only the
   pin moved. That move belongs to platypus#123's own pull request.

## [0.8.0a5] - 2026-10-09

### Added

 - **`Engine.predict_stream`, because `predict` cannot do a large tiled split at all.**
   `predict` returns one stacked array, so the whole split has to be resident: 200 FIVES
   retinas at 2048 is 6.7 GB for the result alone, and the old implementation held **three**
   copies on the way - every tile's probabilities, their concatenation, and the stack of
   stitched images. That killed a run which had finished training (#178).

   The shape of the problem is worth stating, because it is the opposite of what tiling is
   for: `splits` exists for images too large to resize, and the stacked form is least usable
   exactly where `splits` is most needed.

   `predict_stream` yields `(case, prediction)` one source image at a time, stitching only
   the tiles of the image being assembled, so peak memory is one image whatever the split's
   size. The case name travels with the image rather than being left to the caller to line
   up against `samples`: anything that pairs two sequences by position is one `drop_last`
   away from scoring the wrong picture.

   It buffers by example and not by batch, which is the only real risk in the thing - an
   image's tiles can straddle a batch boundary, and a version that stitched each batch would
   mix two pictures. There is a test with four tiles an image and batches of three.

   The model name is resolved before the generator is returned rather than inside it, so a
   mistyped name is an error now instead of on the first `next`, by which time the caller has
   opened the file it was going to write into.

### Changed

 - **`predict` holds one copy instead of three**, since it is now `np.stack` over the stream.
   Same signature, same numbers, same refusal of a partial grid - a test asserts the streamed
   and stacked results are identical element for element. For 200 retinas at 2048 that is
   6.7 GB rather than about 20.
 - **`examples/segment_retinal_vessels.py` calls the public method now.** It had twenty lines
   stepping the model over one sample's tiles by hand, which is what prompted #178: an example
   reaching past a public method says the method is missing something. Both of its callers
   moved over - the clDice pass and the figure, which draws 2 of 200 images and now reads the
   stream for those two instead of asking for all of them at full resolution.

### Fixed

 - **Two DICOM tests depended on a download and reported a network failure as a reader
   bug.** `CT_small.dcm` and `MR_small.dcm` ship inside pydicom; `SC_rgb.dcm` does not - it is
   fetched on first use, and `get_testdata_file` returns `None` when the fetch fails rather
   than raising. Two CI runners hit that on the same commit, and the symptom was
   `could not read 'None'` from inside the reader, which names neither the file nor the
   network. The absence is now named where the file is resolved and the two rows that need it
   skip, because *a test that cannot run and a test that fails are different facts*. Nothing
   about the reader changed; only what the suite says when its data is missing.

### Note

 - **The R half is deliberately not done and the pin stays at `0.8.0a2`** (§3E). Nothing here
   changes a configuration field or an interface R sends, so an R caller is unaffected - but
   this is groundwork R will need rather than a change R does not want: `predict.platypus_fit`
   takes a whole split and nothing else, so platypus#123's vignette cannot score whole images
   from R until R reaches this. That is platypus#123's own R half, not this release's.

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

   **The R half is deliberately not done, and the pin stays at `0.8.0a2`.** This release is an
   example script, a `.gitignore` line and a help string: nothing here is reachable from R, no
   interface moved and no configuration field was added, so there is nothing for the R package
   to call. That leaves the pin three releases behind `main`, which is the mechanism working
   rather than drift - but three is further behind than it has ever been, and `0.8.0a3`'s
   windowed sampler is the first skipped release an R user would actually want, since anyone
   setting `splits` from R pays 13.8 minutes an epoch instead of 9.4 until the pin moves.

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
