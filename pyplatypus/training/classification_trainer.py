"""Training a classifier: the same epoch, a different way of reading it.

`fit` is `Trainer.fit` - the state, the four hooks, the freeze, the history and the stop
reason - and it is written out here rather than inherited for the reason `EngineBase`'s
docstring gives about the engines: what the three trainers share is the *frame*, and what
they do inside it is three different things. `DetectionTrainer.fit` is already 77% of
`Trainer.fit`, measured, so this is the third copy of a loop and the second copy of a
decision not to merge them. What holds them together instead is
`tests/training/test_hook_contract.py`, which runs every trainer the package exports and
requires every hook: `maju116/pyplatypus#300` is what a forgotten hook costs, and that test
exists because this trainer was coming.

**The epoch's own reading is the part that is genuinely different.** A mask metric is a
formula on overlap counts, so an epoch averages what each batch returned, weighted by the
examples in it. Three of this task's four metrics are not means over examples at all -
measured at a prevalence of 3%, where 398 of 500 batches of 8 hold no positive and a
per-class recall, an F1 and a rank statistic are therefore undefined on them - so they are
accumulated across the pass and computed once. The loss is still an example-weighted mean,
which is exactly what it is.

**What this task does not have, and therefore what is not here:** no tiles, so no window and
no stitching; no per-case accumulator, because a case is one image and its accuracy is 0 or
1; no `foreground`, because there is no background class to be the complement of; no source
space to score in, because the answer is a label rather than a picture.

**A label that is a set changes four lines and no structure**, which is the argument for
`multilabel` being one flag on the data rather than a second trainer: the loss is the binary
form of whichever one was asked for, the probabilities are a sigmoid per class, the metrics
hold a two-by-two table each instead of one confusion matrix, and `predict_labels` is a
threshold instead of an `argmax`. Every one of those decisions is taken somewhere else -
`build_loss`, `f.probabilities`, `build_classification_metrics` - and read here. The epoch is
the same epoch. `pyplatypus#307`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pyplatypus.objectives import functional as f
from pyplatypus.objectives.classification import build_classification_metrics
from pyplatypus.objectives.losses import build_loss
from pyplatypus.spec.classification import ClassificationModel
from pyplatypus.training.callbacks import Callback, TrainingState, build_callbacks
from pyplatypus.training.optimizers import build_optimizer, parameter_groups
from pyplatypus.training.trainer import History, format_logs, pick_device


class ClassificationTrainer:
    """One label per image, trained.

    `n_class` is passed rather than read off the specification, the same way
    `build_classifier` takes it: the data decides how many classes there are, and a model
    that also declared it would be a second place for one fact.

    **`multilabel` is passed for the same reason and it is the data's**, not the model's: it
    follows from how the classes were named - one column per class makes each one an
    independent question - and there is nothing in a batch of tensors to tell the two apart,
    because a row of all zeros is ordinary in both worlds. It decides three things and every
    one of them is somewhere else: a sigmoid per class instead of a softmax across them, the
    binary form of whichever loss was asked for, and a two-by-two table per class instead of
    one confusion matrix.
    """

    def __init__(
        self,
        model: nn.Module,
        spec: ClassificationModel,
        *,
        n_class: int,
        multilabel: bool = False,
        device: str | None = None,
        callbacks: list[Callback] | None = None,
    ):
        self.spec = spec
        self.n_class = n_class
        self.multilabel = multilabel
        self.device = pick_device(device)
        self.model = model.to(self.device)
        self.loss_fn = build_loss(spec.loss, multilabel=multilabel)
        self.metrics = build_classification_metrics(spec.metrics, multilabel=multilabel)
        # The layers that arrived pretrained, if any - read exactly as the segmentation
        # trainer reads them, because `encoder_learning_rate` and `freeze_encoder` are on
        # the shared `EncoderModel` and act on the same part.
        self.transferred = getattr(getattr(self.model, "encoder", None), "transferred", None)
        self.optimizer = build_optimizer(
            spec.optimizer,
            parameter_groups(self.model, self.transferred, spec.encoder_learning_rate),
        )
        self.callbacks = callbacks if callbacks is not None else build_callbacks(spec.callbacks)
        self._frozen: bool | None = None

    # ------------------------------------------------------------------- one batch

    def _loss_of(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """The objective, against a **one-hot** target rather than the index the loader sent.

        This package's `cross_entropy` is written for a one-hot target - it is
        `-(target * log_probs).sum(dim=1).mean()`, which is what a mask's target is - and
        handing it an index is not a type error that surfaces. Measured: with a batch size
        different from the class count it raises, and with a batch of 2 against 2 classes
        it returns **1.126928 where the loss is 0.126928**, nine times the value, silently.
        A ragged last batch is exactly how a run reaches that shape.

        So the expansion happens here, in the one place that needs it, rather than in the
        loader - which keeps the index as the thing that crosses, one integer per image
        instead of `n_class` floats, and leaves the metrics free to take either. The control
        that the one-hot route is the right one: on the same logits our `cross_entropy`
        agrees with `torch.nn.functional.cross_entropy`'s index form to every digit.

        **A label that is a set arrives as the target already and is handed over
        untouched**, because there is no index form of a set. It needs no shape check of its
        own: `binary_cross_entropy_with_logits` refuses a `(examples,)` target against
        `(examples, classes)` logits **by name**, including when the batch size equals the
        class count - which is exactly the shape that defeats the categorical form above. So
        the two halves fail differently, and only one of them fails silently.

        **That is true because of the branch and not beside it**, which this paragraph said
        the other way round for an hour. `as_onehot` passes a floating target of the right
        width through unchanged, so the branch looked like a statement of intent that no
        measurement could distinguish - and the mutation that deletes it is caught, by the
        refusal above. Without it a `(3,)` target against `(3, 3)` logits reaches
        `as_onehot`, which reads it as **indices** and expands it into a plausible one-hot,
        and the binary loss then returns 0.843 rather than refusing. So the branch is what
        makes the refusal reachable: the function that rescues the categorical route is the
        same function that would hide the failure on this one.
        """
        if self.multilabel:
            return self.loss_fn(logits, labels)
        return self.loss_fn(logits, f.as_onehot(labels, self.n_class))

    def _set_transferred_frozen(self, frozen: bool) -> None:
        """Freeze or release the transferred layers: `requires_grad` here, `eval()` in
        `_run_epoch`.

        The split is deliberate and is the segmentation trainer's, character for character,
        rather than a third shape for one rule: `model.train(train)` at the top of every
        pass reaches every submodule, so an `eval()` applied here would be undone a moment
        later and the eval half has to be re-applied after that call instead. Both halves
        matter - a frozen BatchNorm in training mode rewrites its running statistics from
        every batch it sees, and pretrained weights were fitted beside pretrained
        statistics, so letting those drift changes the thing being protected.
        """
        if self.transferred is None or frozen == self._frozen:
            return
        for parameter in self.transferred.parameters():
            parameter.requires_grad_(not frozen)
        self._frozen = frozen

    # ------------------------------------------------------------------- one epoch

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        train: bool,
        prefix: str,
        state: TrainingState | None = None,
    ) -> dict[str, float]:
        """One pass: the loss as a mean over examples, the metrics accumulated and computed.

        The metrics are **reset at the start of every pass**, so an epoch's figure is that
        epoch's. Carrying them would make epoch 10 a weighted average of ten epochs and the
        number would improve for no reason the model had anything to do with.
        """
        self.model.train(train)
        # After model.train(), never before: that call reaches every submodule, so a
        # transferred encoder put in eval() earlier would be switched straight back and its
        # BatchNorm statistics would drift after all. The segmentation trainer carries this
        # same three lines and the same comment, and this trainer was written without them -
        # which is `#300`'s shape a third time, caught here by a test that read the
        # transferred module's `training` flag rather than by anybody reviewing it.
        if self._frozen and self.transferred is not None:
            self.transferred.eval()
        for metric in self.metrics.values():
            metric.reset()

        total_loss = 0.0
        examples = 0
        batches = 0

        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self._loss_of(logits, labels)
                if train:
                    loss.backward()
                    self.optimizer.step()
                    # After the step, so a callback reading the weights reads the ones the
                    # step produced - and only on the training pass, because a validation
                    # batch is not a step. `#300` is what the other half of this costs.
                    if state is not None:
                        state.step += 1
                        for callback in self.callbacks:
                            callback.on_batch_end(state)

                size = images.shape[0]
                total_loss += loss.detach().item() * size
                detached = logits.detach()
                for metric in self.metrics.values():
                    metric.update(detached, labels)
                examples += size
                batches += 1

        if batches == 0:
            raise ValueError(f"the {prefix} loader produced no batches")

        epoch = {f"{prefix}_loss": total_loss / examples}
        for name, metric in self.metrics.items():
            # This may refuse, and that is the honest answer rather than a cost to avoid: a
            # per-class figure over a pass holding none of a class is not a number. It is
            # the engine's job to make the splits stratified so that it cannot happen -
            # measured, a random 70/15/15 split of 100 images at 3% prevalence leaves some
            # part empty of the rare class 90% of the time.
            epoch[f"{prefix}_{name}"] = metric.compute()
        return epoch

    # ------------------------------------------------------------------------- fit

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
        *,
        epochs: int | None = None,
        verbose: bool = False,
    ) -> History:
        epochs = epochs if epochs is not None else self.spec.epochs
        state = TrainingState(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=train_loader,
            total_epochs=epochs,
        )
        history = History()

        for callback in self.callbacks:
            callback.on_train_begin(state)

        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            self._set_transferred_frozen(epoch <= self.spec.freeze_encoder)
            logs = self._run_epoch(train_loader, train=True, prefix="train", state=state)
            if validation_loader is not None:
                logs.update(self._run_epoch(validation_loader, train=False, prefix="val"))
            logs["seconds"] = time.perf_counter() - started
            logs["learning_rate"] = self.optimizer.param_groups[0]["lr"]

            state.epoch = epoch
            state.logs = logs
            history.records.append({"epoch": epoch, **logs})
            state.history = history.records
            if verbose:
                print(f"epoch {epoch:>3}  {format_logs(logs)}  ({logs['seconds']:.1f}s)")

            if any(callback.on_epoch_end(state) for callback in self.callbacks):
                history.stop_reason = state.stop_reason
                break

        for callback in self.callbacks:
            callback.on_train_end(state)
        return history

    # -------------------------------------------------------------- scoring, answering

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """One scoring pass.

        No `cases` argument, which is the whole difference from the segmentation trainer's:
        there a case is an image with its own score and a split's figure is the mean over
        cases, so knowing which example is which changes the answer. Here an example *is* a
        case and one image's accuracy is 0 or 1, so the pass is the only unit there is -
        which is also why there is no `val_<metric>_worst`.
        """
        return self._run_epoch(loader, train=False, prefix="val")

    @torch.no_grad()
    def predict_stream(self, loader: DataLoader) -> Iterator[np.ndarray]:
        """One batch of probabilities at a time, so a large set needs no room for all of it.

        Probabilities rather than logits, because that is what a caller can threshold, rank
        or calibrate - and the same quantity `auc` is read off, so a reported score and a
        returned answer cannot disagree about what the model said.

        **Which probabilities is `f.probabilities`' own single line and not a branch here**,
        which is the one place in this package where mutual exclusivity lives: a softmax
        across the classes says they are alternatives and makes them sum to one, a sigmoid
        per class says each is its own yes-or-no. Everything overlap-based on the
        segmentation side already reads that function, and a second copy of the choice here
        is how the two would come to disagree.
        """
        self.model.eval()
        for batch in loader:
            images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(self.device)
            logits = self.model(images)
            yield f.probabilities(logits, multilabel=self.multilabel).cpu().numpy()

    def predict(self, loader: DataLoader) -> np.ndarray:
        """Every probability, as one `(images, classes)` array."""
        return np.concatenate(list(self.predict_stream(loader)), axis=0)

    def predict_labels(
        self, loader: DataLoader, classes: Sequence[str]
    ) -> list[str] | list[list[str]]:
        """The names rather than the indices, because a name is what a reader wanted.

        `classes` is passed rather than read off a spec for the reason `n_class` is: the
        data names the classes, and their **order is the class index** - so handing this the
        wrong list is the one mistake here that produces plausible output, and it is the
        caller's own `data.classes` that is right.

        **A set of labels answers with a list of names per image, and the return type
        therefore depends on the model.** That is uncomfortable in one method and it is the
        honest shape: a name per image would have to pick one of the findings, and the model
        was not asked which finding is *the* one. Nothing here guesses - `multilabel` came
        from the naming of the data this model was trained on.

        The threshold is **0.5 of a probability, which is `logits > 0` exactly** and is the
        same decision `MultiLabelMetric` makes on the hard prediction, so a reported score
        and a returned answer cannot disagree about what the model said. There is no
        threshold field, here or in the specification: a threshold is a number about
        somebody's data, and each class decides for itself.
        """
        if len(classes) != self.n_class:
            raise ValueError(
                f"this model answers over {self.n_class} classes and {len(classes)} names "
                f"were given ({', '.join(classes)}); the names are the data's `classes`, in "
                f"order, because the position is the class index"
            )
        probabilities = self.predict(loader)
        if self.multilabel:
            # In `classes` order rather than by probability, because the position is the
            # class index and a reader comparing two images wants the same order in both.
            return [
                [name for name, chance in zip(classes, row, strict=True) if chance > 0.5]
                for row in probabilities
            ]
        return [classes[index] for index in probabilities.argmax(axis=1)]
