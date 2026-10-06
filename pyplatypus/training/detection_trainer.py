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


class DetectionTrainer:
    def __init__(self, model: nn.Module, spec: DetectionModel, *, anchors,
                 n_class: int, device: str | None = None,
                 callbacks: list[Callback] | None = None, accumulate: int = 1):
        if accumulate < 1:
            raise ValueError(f"accumulate must be at least 1, got {accumulate}")
        self.spec = spec
        self.anchors = anchors
        self.n_class = n_class
        self.accumulate = int(accumulate)
        self.device = pick_device(device)
        self.model = model.to(self.device)
        self.loss_fn = Yolo3Loss(
            anchors=anchors, n_class=n_class,
            input_shape=(int(spec.input_shape[0]), int(spec.input_shape[1])),
            ignore_threshold=spec.ignore_threshold,
            box_loss=spec.box_loss,
        )
        self.optimizer = build_optimizer(
            spec.optimizer, parameter_groups(self.model, None, None)
        )
        self.callbacks = callbacks if callbacks is not None else build_callbacks(
            spec.callbacks
        )

    # ------------------------------------------------------------------ one epoch
    def _run_epoch(self, loader: DataLoader, *, train: bool, prefix: str
                   ) -> dict[str, float]:
        self.model.train(train)
        totals: dict[str, float] = {}
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
                        self._step()
                        pending = 0

                for key, value in parts.as_dict().items():
                    totals[f"{prefix}_{key}"] = totals.get(f"{prefix}_{key}", 0.0) + value
                batches += 1

        if train and pending:
            # Whatever is left at the end of an epoch, rather than discarded: the last
            # group is usually partial.
            self._step()
        if batches == 0:
            raise ValueError(f"the {prefix} loader produced no batches")
        return {key: value / batches for key, value in totals.items()}

    def _step(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ fit
    def fit(self, train_loader: DataLoader, validation_loader: DataLoader | None = None,
            *, epochs: int | None = None, verbose: bool = False) -> History:
        epochs = epochs if epochs is not None else self.spec.epochs
        state = TrainingState(model=self.model, optimizer=self.optimizer,
                              train_loader=train_loader,
                              total_epochs=epochs)
        history = History()

        for callback in self.callbacks:
            callback.on_train_begin(state)

        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            logs = self._run_epoch(train_loader, train=True, prefix="train")
            if validation_loader is not None:
                logs.update(self._run_epoch(validation_loader, train=False, prefix="val"))
            logs["seconds"] = time.perf_counter() - started
            logs["learning_rate"] = self.optimizer.param_groups[0]["lr"]

            state.epoch = epoch
            state.logs = logs
            history.records.append({"epoch": epoch, **logs})
            state.history = history.records
            if verbose:
                print(f"epoch {epoch:>3}  {format_logs(logs)}  "
                      f"({logs['seconds']:.1f}s)")

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
