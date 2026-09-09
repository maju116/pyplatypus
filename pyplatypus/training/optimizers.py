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
        p, lr=s.learning_rate, betas=(s.beta_1, s.beta_2), eps=s.eps,
        weight_decay=s.weight_decay, amsgrad=s.amsgrad),
    AdamW: lambda p, s: torch.optim.AdamW(
        p, lr=s.learning_rate, betas=(s.beta_1, s.beta_2), eps=s.eps,
        weight_decay=s.weight_decay),
    Sgd: lambda p, s: torch.optim.SGD(
        p, lr=s.learning_rate, momentum=s.momentum, nesterov=s.nesterov,
        weight_decay=s.weight_decay),
    RmsProp: lambda p, s: torch.optim.RMSprop(
        p, lr=s.learning_rate, alpha=s.alpha, eps=s.eps, momentum=s.momentum,
        weight_decay=s.weight_decay),
    Adagrad: lambda p, s: torch.optim.Adagrad(
        p, lr=s.learning_rate, lr_decay=s.lr_decay, eps=s.eps,
        weight_decay=s.weight_decay),
    Adadelta: lambda p, s: torch.optim.Adadelta(
        p, lr=s.learning_rate, rho=s.rho, eps=s.eps, weight_decay=s.weight_decay),
    Adamax: lambda p, s: torch.optim.Adamax(
        p, lr=s.learning_rate, betas=(s.beta_1, s.beta_2), eps=s.eps,
        weight_decay=s.weight_decay),
    NAdam: lambda p, s: torch.optim.NAdam(
        p, lr=s.learning_rate, betas=(s.beta_1, s.beta_2), eps=s.eps,
        weight_decay=s.weight_decay),
}


def build_optimizer(spec: OptimizerSpec,
                    parameters: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
    builder = _BUILDERS.get(type(spec))
    if builder is None:  # pragma: no cover - unreachable while the union is exhaustive
        raise KeyError(f"no implementation for optimizer {type(spec).__name__}")
    return builder(parameters, spec)
