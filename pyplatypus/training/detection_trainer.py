"""Training a detector.

Separate from `Trainer` rather than a branch inside it, because almost everything the
segmentation trainer does is specific to a mask. It takes one target tensor; this takes
three. It computes a metric every batch; mean average precision cannot be computed from a
batch at all - it needs the whole split decoded, suppressed and matched, so there is
nothing to accumulate per step. It picks a loss from a menu; YOLOv3's objective is part of
the architecture.

What they do share is kept shared: `History`, `pick_device`, the optimizer builder and the
callbacks, which watch `train_loss` and `val_loss` and so work here unchanged.

The one thing worth saying about the loss is that **it does not reach zero and that is not
a stall**. Binary cross-entropy against a soft target bottoms out at the target's own
entropy, so the coordinate term has a floor above zero that depends on the data. The four
parts are reported separately for exactly that reason: a total of 5.94 means nothing alone,
and `coordinates 5.93 / objectness 0.001 / no_object 0.004 / classes 0.000` says the model
has learned what is where and is still refining the boxes.
"""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from pyplatypus.detection.loss import Yolo3Loss
from pyplatypus.spec.detection import DetectionModel
from pyplatypus.training.callbacks import Callback, TrainingState, build_callbacks
from pyplatypus.training.optimizers import build_optimizer, parameter_groups
from pyplatypus.training.trainer import History, format_logs, pick_device

#: Gradients are clipped to this norm. YOLOv3's loss is a sum of four terms over three
#: grids, and early in training the no-object term dominates by orders of magnitude; one
#: batch of an unlucky scale can otherwise undo an epoch. Measured during the BCCD run:
#: without it the loss rose from 9.1 to 54.6 before recovering.
GRADIENT_CLIP = 10.0


#: The four terms an epoch pools, in the order `LossParts` declares them. `loss` is not
#: among them: it is their weighted sum and is rebuilt from them rather than pooled.
TERMS = ("coordinates", "objectness", "no_object", "classes")


class DetectionTrainer:
    def __init__(
        self,
        model: nn.Module,
        spec: DetectionModel,
        *,
        anchors,
        n_class: int,
        device: str | None = None,
        callbacks: list[Callback] | None = None,
        accumulate: int = 1,
    ):
        if accumulate < 1:
            raise ValueError(f"accumulate must be at least 1, got {accumulate}")
        self.spec = spec
        self.anchors = anchors
        self.n_class = n_class
        self.accumulate = int(accumulate)
        self.device = pick_device(device)
        self.model = model.to(self.device)
        self.loss_fn = Yolo3Loss(
            anchors=anchors,
            n_class=n_class,
            input_shape=(int(spec.input_shape[0]), int(spec.input_shape[1])),
            ignore_threshold=spec.ignore_threshold,
            box_loss=spec.box_loss,
        )
        self.optimizer = build_optimizer(spec.optimizer, parameter_groups(self.model, None, None))
        self.callbacks = callbacks if callbacks is not None else build_callbacks(spec.callbacks)

    # ------------------------------------------------------------------ one epoch
    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        train: bool,
        prefix: str,
        state: TrainingState | None = None,
    ) -> dict[str, float]:
        self.model.train(train)
        # Per grid and per term: the un-normalised sum, and what it was divided by. An
        # epoch is then `sum_g (sum_g / count_g)` for each term - the arithmetic the loss
        # does within a batch, carried across the epoch - rather than a mean over batches.
        # `maju116/rplatypus#218`.
        weighted: list[dict[str, float]] = []
        divisors: list[dict[str, float]] = []
        batches = 0
        pending = 0
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = [t.to(self.device, non_blocking=True) for t in targets]
                parts = self.loss_fn(self.model(images), targets)

                if train:
                    # Divided by the accumulation so the gradient is the mean over the
                    # effective batch rather than its sum; otherwise the learning rate
                    # would mean something different at every setting.
                    (parts.total / self.accumulate).backward()
                    pending += 1
                    if pending == self.accumulate:
                        self._step(state)
                        pending = 0

                if not parts.grids:
                    raise ValueError(
                        "the loss reported no per-grid parts, so an epoch cannot be pooled "
                        "by the counts it divided by"
                    )
                for index, grid in enumerate(parts.grids):
                    while len(weighted) <= index:
                        weighted.append(dict.fromkeys(TERMS, 0.0))
                        divisors.append(dict.fromkeys(TERMS, 0.0))
                    for term in TERMS:
                        denominator = grid.cells if term == "no_object" else grid.objects
                        weighted[index][term] += getattr(grid, term) * denominator
                        divisors[index][term] += denominator
                batches += 1

        if train and pending:
            # Whatever is left at the end of an epoch, rather than discarded: the last
            # group is usually partial.
            self._step(state)
        if batches == 0:
            raise ValueError(f"the {prefix} loader produced no batches")

        # The grids are SUMMED, because that is what `forward` does with them: a term is
        # `sum_g (part_g)` where each `part_g` already carries its own grid's divisor. A
        # pooling that mixed the grids would turn a sum of three into an average of three,
        # which is a different quantity rather than a different weighting - it moves the
        # reported loss by a factor of about three and would pick epochs for that reason.
        epoch = {}
        for term in TERMS:
            value = 0.0
            for grid in range(len(weighted)):
                divisor = divisors[grid][term]
                # A grid that saw nothing to divide by contributed a zero term, so it
                # contributes nothing here either rather than a division by zero.
                if divisor:
                    value += weighted[grid][term] / divisor
            epoch[f"{prefix}_{term}"] = value

        # `total` is reconstructed rather than pooled. It is a WEIGHTED SUM of four terms
        # divided by two different denominators, so it has no denominator of its own and
        # nothing to pool it by; the loss builds it the same way from the same weights.
        epoch[f"{prefix}_loss"] = sum(
            self.loss_fn.weights[term] * epoch[f"{prefix}_{term}"] for term in TERMS
        )
        return epoch

    def _step(self, state: TrainingState | None = None) -> None:
        """One optimiser step, and the fourth callback hook.

        **The hook lives here rather than at either call site**, and there are two of them -
        a full accumulation group inside the loop and whatever is left at the end of the
        epoch - so a third one could not forget it. `maju116/pyplatypus#300`: this trainer
        called three of the four hooks, `ema` is the only callback that implements the
        fourth, and a detection run configured with `ema` therefore took no average at all.
        Measured over four batches and two epochs, against the same callback through the
        segmentation trainer: 0 folds here against 8 there.

        **It fires per STEP and not per batch, which is only a distinction when
        `accumulate` is above 1 - and there it is the difference between `decay` meaning
        one thing and meaning a different thing at every setting.** Measured on a weight
        that moves by 1.0 per step, decay 0.9, 32 batches: at `accumulate: 1` the two
        readings are identical to the digit, 23.3434; at `accumulate: 4` folding per step
        gives 3.3047 and per batch 5.4728, against a true weight of 8.0. Per batch leaves
        3.43e-02 of the initial value whatever the accumulation, where per step leaves
        4.30e-01 at 4 - so the horizon would shrink by exactly the accumulation. That is
        the same objection this file already makes about dividing the gradient by
        `accumulate`.

        And after the step rather than before, so a callback reading the weights reads the
        ones the step produced - the segmentation trainer's own words for the same hook.
        """
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if state is not None:
            state.step += 1
            for callback in self.callbacks:
                callback.on_batch_end(state)

    # ------------------------------------------------------------------ fit
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

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """The loss on a split, part by part.

        Not the thing to report - mean average precision is - but the thing a callback
        watches, and the only number available while an epoch is running.
        """
        return self._run_epoch(loader, train=False, prefix="val")

    @torch.no_grad()
    def raw_outputs(self, image: torch.Tensor) -> list:
        """The three grids of logits for one image, as numpy, ready for `decode`.

        One image at a time on purpose: decoding, suppressing and matching happen per
        image, and batching here would buy little while making the index bookkeeping a
        place for an off-by-one to hide.
        """
        self.model.eval()
        outputs = self.model(image[None].to(self.device))
        return [out[0].cpu().numpy() for out in outputs]
