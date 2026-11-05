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


# --- the transferred group, whose body no test had ever executed --------------------------
#
# `maju116/pyplatypus#395`. `parameter_groups` returns `model.parameters()` when there is
# nothing to split, and the eleventh review pass found that the **other** branch - the one
# that builds `{"params": theirs, "lr": transferred_rate}` - was marked UNEXECUTED by
# coverage over the whole suite. So `encoder_learning_rate`, which is the only route to a
# separate rate for a transferred backbone, reached nothing that any test ran.
#
# Two levels, because `#395`'s own lesson is that a guard one level below the change is a
# guard the change can be removed under: the function here, and the two trainers that pass
# the spec's field into it.


def _split_model():
    """A model with an identifiable transferred part, which is what the function keys on."""
    transferred = torch.nn.Linear(4, 4)
    model = torch.nn.Sequential(transferred, torch.nn.Linear(4, 2))
    return model, transferred


def test_a_transferred_rate_builds_two_groups_and_only_one_carries_it():
    from pyplatypus.training.optimizers import parameter_groups

    model, transferred = _split_model()
    groups = parameter_groups(model, transferred, 1e-5)

    assert isinstance(groups, list) and len(groups) == 2
    with_rate = [g for g in groups if "lr" in g]
    without = [g for g in groups if "lr" not in g]
    assert len(with_rate) == 1 and len(without) == 1
    assert with_rate[0]["lr"] == 1e-5

    # Identity decides the membership, not names - which is what the docstring claims and
    # what no renaming can get wrong.
    theirs = {id(p) for p in transferred.parameters()}
    assert {id(p) for p in with_rate[0]["params"]} == theirs
    assert {id(p) for p in without[0]["params"]}.isdisjoint(theirs)


def test_the_built_optimiser_really_steps_the_two_groups_at_different_rates():
    """The reason the branch exists, measured rather than asserted about the dict: one
    gradient of 1.0 everywhere has to move the two halves by different amounts."""
    from pyplatypus.spec.components import Sgd
    from pyplatypus.training.optimizers import parameter_groups

    model, transferred = _split_model()
    optimiser = build_optimizer(Sgd(learning_rate=0.1), parameter_groups(model, transferred, 0.001))

    before = [p.detach().clone() for p in model.parameters()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimiser.step()

    moved = [float((p.detach() - was).abs().max()) for p, was in zip(model.parameters(), before)]
    transferred_ids = {id(p) for p in transferred.parameters()}
    theirs = [m for p, m in zip(model.parameters(), moved) if id(p) in transferred_ids]
    ours = [m for p, m in zip(model.parameters(), moved) if id(p) not in transferred_ids]

    # The bound is between two measured numbers rather than chosen, which is `CLAUDE.md`
    # §4's rule - and the first version of this line said `abs=1e-9`, **below the noise**,
    # so it failed on correct code. Measured over 50 builds: the worst float32 deviation is
    # 1.29e-08 on the transferred group and 2.38e-08 on ours, while ignoring the rate moves
    # the transferred group by 0.099. So 1e-6 sits about 42 times above the noise and five
    # orders of magnitude below the defect.
    assert all(m == pytest.approx(0.001, abs=1e-6) for m in theirs)
    assert all(m == pytest.approx(0.1, abs=1e-6) for m in ours)


def test_no_transferred_part_takes_the_path_it_always_took():
    """The control: without a backbone the function must return the plain parameters, so the
    two assertions above are about the branch rather than about any list of dicts."""
    from pyplatypus.training.optimizers import parameter_groups

    model, transferred = _split_model()
    assert not isinstance(parameter_groups(model, None, 1e-5), list)
    assert not isinstance(parameter_groups(model, transferred, None), list)
