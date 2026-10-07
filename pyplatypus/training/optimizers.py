"""Spec to torch optimiser. Nothing clever; the spec already validated the ranges."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from pyplatypus.spec.components import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    NAdam,
    OptimizerSpec,
    RmsProp,
    Sgd,
)

_BUILDERS = {
    Adam: lambda p, s: torch.optim.Adam(
        p,
        lr=s.learning_rate,
        betas=(s.beta_1, s.beta_2),
        eps=s.eps,
        weight_decay=s.weight_decay,
        amsgrad=s.amsgrad,
    ),
    AdamW: lambda p, s: torch.optim.AdamW(
        p, lr=s.learning_rate, betas=(s.beta_1, s.beta_2), eps=s.eps, weight_decay=s.weight_decay
    ),
    Sgd: lambda p, s: torch.optim.SGD(
        p, lr=s.learning_rate, momentum=s.momentum, nesterov=s.nesterov, weight_decay=s.weight_decay
    ),
    RmsProp: lambda p, s: torch.optim.RMSprop(
        p,
        lr=s.learning_rate,
        alpha=s.alpha,
        eps=s.eps,
        momentum=s.momentum,
        weight_decay=s.weight_decay,
    ),
    Adagrad: lambda p, s: torch.optim.Adagrad(
        p, lr=s.learning_rate, lr_decay=s.lr_decay, eps=s.eps, weight_decay=s.weight_decay
    ),
    Adadelta: lambda p, s: torch.optim.Adadelta(
        p, lr=s.learning_rate, rho=s.rho, eps=s.eps, weight_decay=s.weight_decay
    ),
    Adamax: lambda p, s: torch.optim.Adamax(
        p, lr=s.learning_rate, betas=(s.beta_1, s.beta_2), eps=s.eps, weight_decay=s.weight_decay
    ),
    NAdam: lambda p, s: torch.optim.NAdam(
        p, lr=s.learning_rate, betas=(s.beta_1, s.beta_2), eps=s.eps, weight_decay=s.weight_decay
    ),
}


def build_optimizer(
    spec: OptimizerSpec, parameters: Iterable[nn.Parameter]
) -> torch.optim.Optimizer:
    builder = _BUILDERS.get(type(spec))
    if builder is None:  # pragma: no cover - unreachable while the union is exhaustive
        raise KeyError(f"no implementation for optimizer {type(spec).__name__}")
    return builder(parameters, spec)


def parameter_groups(
    model: nn.Module, transferred: nn.Module | None, transferred_rate: float | None
) -> list[dict] | Iterable[nn.Parameter]:
    """Split a model into a transferred group and the rest, when the two want
    different learning rates.

    Returns the plain parameters when there is nothing to split, so a model with no
    pretrained part takes exactly the path it always took. Identity is what separates the
    groups, not names: a parameter is in the transferred group because it is one of that
    module's, which no renaming can get wrong.
    """
    if transferred is None or transferred_rate is None:
        return model.parameters()

    inside = {id(p) for p in transferred.parameters()}
    theirs = [p for p in model.parameters() if id(p) in inside]
    ours = [p for p in model.parameters() if id(p) not in inside]
    groups = []
    if ours:
        groups.append({"params": ours})
    if theirs:
        groups.append({"params": theirs, "lr": transferred_rate})
    return groups
