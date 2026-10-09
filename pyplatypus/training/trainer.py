"""Training one model.

Deep supervision is handled here rather than in the losses: the model hands back one
prediction per depth, the loss is averaged over all of them, and the metrics are scored
on the final one only - which is the prediction the user will actually receive.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pyplatypus.data.images import stitch
from pyplatypus.objectives import build_loss, build_metrics
from pyplatypus.objectives import functional as f
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.training.callbacks import Callback, TrainingState, build_callbacks
from pyplatypus.training.optimizers import build_optimizer, parameter_groups


@dataclass
class History:
    """One row per epoch. Plain data, so it crosses into R as a data.frame."""

    records: list[dict[str, float]] = field(default_factory=list)
    stop_reason: str | None = None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def columns(self) -> list[str]:
        return list(self.records[0]) if self.records else []

    def best(self, key: str = "val_loss") -> dict[str, float] | None:
        if not self.records or key not in self.records[0]:
            return None
        lower_is_better = key.endswith("loss")
        return (min if lower_is_better else max)(self.records, key=lambda r: r[key])

    def to_dict(self) -> dict[str, Any]:
        return {"records": self.records, "stop_reason": self.stop_reason}


def seed_everything(seed: int | None) -> None:
    """Make a run reproducible, as far as it can be made reproducible.

    `seed` has been a field on the specification since the first release, described as
    "set it if you want a reproducible run", and **nothing read it**. A flag that does
    nothing is worse than a missing one: in R, `platypus_spec(seed = 1)` made a promise
    and two runs of it disagreed. Found while wiring detection in, where the anchors are
    fitted by k-means and the seed had to come from somewhere.

    What this does not do is ask torch for deterministic algorithms. That makes some
    convolutions much slower and makes others raise, and the result would be a seed that
    sometimes refuses to run at all. Two runs at one seed on one machine agree; across
    machines, or across a cuDNN version, they need not.
    """
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed)
    # Seeds every device, and the generator DataLoader derives its workers' seeds from,
    # so augmentation in a worker process is reproducible too.
    torch.manual_seed(seed)


def format_logs(logs: dict[str, float]) -> str:
    """One epoch's numbers, for a person watching.

    `learning_rate` gets its own format because four decimal places made the whole point
    of a schedule invisible: a cosine decaying 1e-4 to 1e-8 printed `learning_rate=0.0000`
    from epoch 90 onwards, so a run with a schedule and a run without looked identical.
    Found by watching one, which is the only way this kind of thing is found.
    """
    parts = []
    for key, value in logs.items():
        if key == "seconds":
            continue
        parts.append(f"{key}={value:.3g}" if key == "learning_rate" else f"{key}={value:.4f}")
    return " ".join(parts)


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        spec: SegmentationModel,
        *,
        device: str | None = None,
        callbacks: list[Callback] | None = None,
    ):
        self.spec = spec
        self.device = pick_device(device)
        self.model = model.to(self.device)
        self.loss_fn = build_loss(spec.loss)
        self.metrics = build_metrics(spec.metrics)
        # The layers that arrived pretrained, if any. Held separately because both
        # `encoder_learning_rate` and `freeze_encoder` act on exactly that part and on
        # nothing else - the full-resolution stage in front of them is ours and random.
        self.transferred = getattr(getattr(self.model, "encoder", None), "transferred", None)
        self.optimizer = build_optimizer(
            spec.optimizer,
            parameter_groups(self.model, self.transferred, spec.encoder_learning_rate),
        )
        self.callbacks = callbacks if callbacks is not None else build_callbacks(spec.callbacks)
        self._frozen: bool | None = None
        self._wants_distance = getattr(self.loss_fn, "needs_distance", False)

    def _loss_and_final(
        self, batch_x: torch.Tensor, batch_y: torch.Tensor, distance: torch.Tensor | None = None
    ):
        out = self.model(batch_x)
        # The flag, not the value: a loss that does not want a distance map must not be
        # handed one, and eight of the nine take two arguments. Read with a default
        # because `loss_fn` need only be callable - the NaN test substitutes a bare
        # function - and something that does not say it wants a distance map does not.
        call = (
            (lambda o: self.loss_fn(o, batch_y, distance))
            if self._wants_distance
            else (lambda o: self.loss_fn(o, batch_y))
        )
        if isinstance(out, tuple):
            # Deep supervision: every depth is trained, the deepest is reported.
            return torch.stack([call(o) for o in out]).mean(), out[-1]
        return call(out), out

    def _run_epoch(self, loader: DataLoader, *, train: bool, prefix: str) -> dict[str, float]:
        self.model.train(train)
        # After model.train(), never before: that call reaches every submodule, so a
        # transferred encoder put in eval() earlier would be switched straight back and
        # its BatchNorm statistics would drift after all.
        if self._frozen and self.transferred is not None:
            self.transferred.eval()
        totals: dict[str, float] = {f"{prefix}_loss": 0.0}
        totals.update({f"{prefix}_{name}": 0.0 for name in self.metrics})
        batches = 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                # Two items or three: the third is the signed distance map, which the
                # loader produces only when the loss asked for it. Unpacked by length
                # rather than by a flag, so a loader and a trainer cannot disagree.
                batch_x, batch_y = batch[0], batch[1]
                distance = batch[2].to(self.device, non_blocking=True) if len(batch) > 2 else None
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                loss, final = self._loss_and_final(batch_x, batch_y, distance)
                if train:
                    loss.backward()
                    self.optimizer.step()
                # detach before reading: torch >= 2.14 warns about turning a tensor
                # that still tracks gradients into a scalar, and it would warn once per
                # batch for the whole run.
                totals[f"{prefix}_loss"] += loss.detach().item()
                for name, metric in self.metrics.items():
                    totals[f"{prefix}_{name}"] += metric(final, batch_y).detach().item()
                batches += 1

        if batches == 0:
            raise ValueError(f"the {prefix} loader produced no batches")
        return {key: value / batches for key, value in totals.items()}

    def _set_transferred_frozen(self, frozen: bool) -> None:
        """Freeze or release the transferred layers.

        `eval()` as well as `requires_grad`, and that second half is the one that is easy
        to miss: a frozen BatchNorm still rewrites its running mean and variance from
        every batch it sees. Pretrained weights were fitted alongside pretrained
        statistics, so letting the statistics drift while holding the weights fixed
        changes the thing being protected, quietly and from the first batch.
        """
        if self.transferred is None or self._frozen == frozen:
            return
        for parameter in self.transferred.parameters():
            parameter.requires_grad_(not frozen)
        self._frozen = frozen

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
            logs = self._run_epoch(train_loader, train=True, prefix="train")
            if validation_loader is not None:
                logs.update(self._run_epoch(validation_loader, train=False, prefix="val"))
            logs["seconds"] = time.perf_counter() - started
            logs["learning_rate"] = self.optimizer.param_groups[0]["lr"]

            state.epoch = epoch
            state.logs = logs
            record = {"epoch": epoch, **logs}
            history.records.append(record)
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
        return self._run_epoch(loader, train=False, prefix="val")

    @torch.no_grad()
    def score_cases(self, loader: DataLoader, cases: Sequence[str]) -> list[dict[str, Any]]:
        """Metrics for each case separately, rather than one number for the whole split.

        `cases` names the case each example belongs to, in the order the loader serves
        them - so a tiled image contributes several examples under one name.

        Tiles are summed, not averaged. Dice is a ratio of sums, so adding up TP, FP and FN
        across an image's tiles and applying the formula once gives that image's Dice
        exactly; averaging the tiles' Dice scores gives a different number, and the
        difference is largest where it matters most - a tile holding a sliver of the object
        scores badly and drags down a case the model actually segmented well.

        Why per case at all: a single figure over a split hides the distribution, and the
        distribution is the finding. "Dice 0.86" and "Dice 0.86, but 0.2 on three of the
        forty patients" are not the same result, and only one of them is honest.
        """
        self.model.eval()
        if len(cases) != len(loader.dataset):
            raise ValueError(
                f"{len(cases)} case names for {len(loader.dataset)} examples; they must "
                "line up one to one"
            )

        totals: dict[str, list[torch.Tensor]] = {}
        direct: dict[str, dict[str, list[torch.Tensor]]] = {}
        order: list[str] = []
        position = 0

        # By length, as `_run_epoch` does: a loader built elsewhere may carry a distance
        # map this does not need, and crashing on an extra item nobody reads would be a
        # poor way to say so.
        for batch in loader:
            batch_x = batch[0].to(self.device, non_blocking=True)
            batch_y = batch[1].to(self.device, non_blocking=True)
            out = self.model(batch_x)
            if isinstance(out, tuple):
                out = out[-1]
            hard = f.as_onehot(out.argmax(dim=1), out.shape[1])
            tp, fp, fn = f.overlaps(hard, batch_y)
            # A metric that reads the shape of a whole mask cannot be rebuilt from overlap
            # counts, so it is computed here, per example, while the mask still exists.
            whole = {
                name: metric.coefficient(hard, batch_y)
                for name, metric in self.metrics.items()
                if not metric.accumulates
            }

            for offset in range(batch_x.shape[0]):
                case = cases[position + offset]
                if case not in totals:
                    totals[case] = [torch.zeros_like(tp[offset]) for _ in range(3)]
                    order.append(case)
                for slot, value in enumerate((tp, fp, fn)):
                    totals[case][slot] += value[offset]
                for name, value in whole.items():
                    direct.setdefault(name, {}).setdefault(case, []).append(value[offset])
            position += batch_x.shape[0]

        rows = []
        for case in order:
            case_tp, case_fp, case_fn = totals[case]
            row: dict[str, Any] = {"case": case}
            for name, metric in self.metrics.items():
                if metric.accumulates:
                    row[name] = metric.reduce(metric.combine(case_tp, case_fp, case_fn)).item()
                    continue
                pieces = direct[name][case]
                if len(pieces) > 1:
                    raise ValueError(
                        f"'{name}' reads the shape of a whole mask and this case arrived in "
                        f"{len(pieces)} pieces, which it cannot be rebuilt from. A tiled run "
                        "is refused when the specification is read; a case in pieces here "
                        "means something else split it."
                    )
                row[name] = metric.reduce(pieces[0]).item()
            rows.append(row)
        return rows

    @torch.no_grad()
    def predict_stream(self, loader: DataLoader) -> Iterator[np.ndarray]:
        """One source image's class probabilities at a time, channels-last, tiles stitched.

        The old package stopped at the tiles. Cutting an HD image into a grid is only
        useful if what comes back is the same size as what went in, so if the model tiles,
        consecutive tiles are stitched into one mask per source image here.

        Nothing is held but the tiles of the image being assembled, so peak memory is one
        source image whatever the split's size. That is the difference between this and
        `predict`: tiling exists for images too large to resize, and the stacked form then
        needs the whole split resident at full resolution - 6.7 GB for 200 FIVES retinas,
        which is what killed a run that had finished training (#178).
        """
        self.model.eval()
        splits = self.spec.splits
        per_image = 1 if splits is None else self.spec.tiles_per_image
        buffer: list[np.ndarray] = []
        for batch in loader:
            batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
            out = self.model(batch_x.to(self.device))
            if isinstance(out, tuple):
                out = out[-1]
            probabilities = out.softmax(dim=1).cpu().numpy()
            # One array per example rather than per batch: an image's tiles can straddle a
            # batch boundary, so the buffer counts examples and not batches.
            for tile in np.moveaxis(probabilities, 1, -1):
                buffer.append(tile)
                if len(buffer) == per_image:
                    # A copy, not the view `moveaxis` handed over: yielding the view
                    # would keep the whole batch alive for as long as the caller holds
                    # one image, which is the opposite of the point.
                    stacked = np.stack(buffer)
                    yield stacked[0].copy() if splits is None else stitch(stacked, splits)
                    buffer.clear()
        if buffer:
            raise ValueError(
                f"{len(buffer)} tiles left over, which is not a whole number of images "
                f"at {per_image} tiles each; the loader must not drop or shuffle them"
            )

    def predict(self, loader: DataLoader) -> np.ndarray:
        """Class probabilities for every example, channels-last, tiles reassembled.

        One stacked array, so every prediction must fit in memory at once. For a tiled
        split that is the whole thing at full resolution; `predict_stream` is the form that
        does not hold it, and the engine's `predict_stream` is how a caller reaches it.
        """
        return np.stack(list(self.predict_stream(loader)))
