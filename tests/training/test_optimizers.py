import pytest
import torch

from pyplatypus.spec.components import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    NAdam,
    RmsProp,
    Sgd,
)
from pyplatypus.training import build_optimizer

ALL = [Adam(), AdamW(), Sgd(), RmsProp(), Adagrad(), Adadelta(), Adamax(), NAdam()]


@pytest.mark.parametrize("spec", ALL, ids=[s.name for s in ALL])
def test_every_optimiser_builds_and_steps(spec):
    parameter = torch.nn.Parameter(torch.ones(4))
    optimiser = build_optimizer(spec, [parameter])
    (parameter.sum() * 2).backward()
    before = parameter.detach().clone()
    optimiser.step()
    assert not torch.equal(before, parameter.detach())


def test_learning_rate_reaches_the_optimiser():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimiser = build_optimizer(Adam(learning_rate=0.123), [parameter])
    assert optimiser.param_groups[0]["lr"] == pytest.approx(0.123)


def test_sgd_momentum_and_nesterov_reach_the_optimiser():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimiser = build_optimizer(Sgd(momentum=0.9, nesterov=True), [parameter])
    assert optimiser.param_groups[0]["momentum"] == pytest.approx(0.9)
    assert optimiser.param_groups[0]["nesterov"] is True
