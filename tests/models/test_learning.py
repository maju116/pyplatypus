"""Can these networks actually learn?

Shape tests pass for a model that is wired wrongly - a decoder ignoring its skips still
returns the right shape. So each architecture is made to overfit one tiny synthetic
image. If Dice does not reach near-perfect on a problem this easy, something is broken
in a way no shape assertion would show.

torch's own CrossEntropyLoss is used deliberately: this is a test of the architectures,
not of our loss functions, which arrive in step 4.
"""

import pytest
import torch
from torch import nn

from pyplatypus.models import build_model
from pyplatypus.spec.common import Architecture
from pyplatypus.spec.models import SegmentationModel


def synthetic(size=32, rank=2, seed=0):
    """A filled circle (or ball) on noise. Trivial, and that is the point."""
    torch.manual_seed(seed)
    coords = torch.meshgrid(*[torch.arange(size, dtype=torch.float32)] * rank, indexing="ij")
    centre = (size - 1) / 2
    distance = sum((c - centre) ** 2 for c in coords).sqrt()
    target = (distance < size / 4).long()
    image = target.float() * 0.8 + torch.rand_like(target, dtype=torch.float32) * 0.2
    return image[None, None], target[None]


def overfit(model, image, target, steps=60, lr=1e-2):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    first = None
    for _ in range(steps):
        optimiser.zero_grad()
        out = model(image)
        if isinstance(out, tuple):
            loss = sum(criterion(o, target) for o in out) / len(out)
        else:
            loss = criterion(out, target)
        loss.backward()
        optimiser.step()
        if first is None:
            first = float(loss)
    return first, float(loss)


def dice(model, image, target):
    with torch.no_grad():
        out = model(image)
        out = out[-1] if isinstance(out, tuple) else out
        predicted = out.argmax(1)
    overlap = (predicted * target).sum()
    return float(2 * overlap / (predicted.sum() + target.sum() + 1e-8))


@pytest.mark.parametrize("architecture", list(Architecture))
def test_each_architecture_can_overfit_one_image(architecture):
    image, target = synthetic()
    model = build_model(SegmentationModel(
        name="m", architecture=architecture, input_shape=(32, 32), channels=1,
        blocks=2, filters=8,
    ), n_class=2)
    first, last = overfit(model, image, target)
    assert last < first, f"{architecture.value} did not reduce its loss at all"
    assert dice(model, image, target) > 0.95, f"{architecture.value} failed to fit"


def test_deep_supervision_learns_too():
    image, target = synthetic()
    model = build_model(SegmentationModel(
        name="m", architecture=Architecture.U_NET_PLUS_PLUS, input_shape=(32, 32),
        channels=1, blocks=2, filters=8, deep_supervision=True,
    ), n_class=2)
    first, last = overfit(model, image, target)
    assert last < first
    assert dice(model, image, target) > 0.95


def test_a_3d_model_can_overfit_one_volume():
    """The rank-generic builder has to produce something that trains, not merely
    something that has the right shape."""
    image, target = synthetic(size=16, rank=3)
    model = build_model(SegmentationModel(
        name="m", input_shape=(16, 16, 16), channels=1, blocks=2, filters=8,
    ), n_class=2)
    first, last = overfit(model, image, target, steps=80)
    assert last < first
    assert dice(model, image, target) > 0.9


def test_separable_convolutions_still_learn():
    image, target = synthetic()
    model = build_model(SegmentationModel(
        name="m", input_shape=(32, 32), channels=1, blocks=2, filters=8,
        separable_conv=True,
    ), n_class=2)
    first, last = overfit(model, image, target, steps=100)
    assert last < first
    assert dice(model, image, target) > 0.9


def test_skips_are_actually_used():
    """Cut the skip connections and the same budget should do measurably worse.

    Getting this test to mean anything took three attempts, and the first two are worth
    recording because both passed while testing nothing.

    Overfitting one image needs no skips at all: the bottleneck holds 8x8x64 numbers,
    which is more than enough to memorise a 64x64 mask regardless of what came in. Making
    the target finer did not help either - it was still one image, still memorised.

    So the model has to *generalise*. Here the target is a per-pixel function of the
    input, evaluated on images the model never saw. That detail exists only at full
    resolution: it reaches the output through the skip connections or not at all.
    """
    torch.manual_seed(0)
    spec = SegmentationModel(name="m", input_shape=(32, 32), channels=1, 
                             blocks=3, filters=8)

    def batch(n, seed):
        generator = torch.Generator().manual_seed(seed)
        image = torch.rand(n, 1, 32, 32, generator=generator)
        return image, (image[:, 0] > 0.5).long()

    train_x, train_y = batch(8, seed=1)
    test_x, test_y = batch(4, seed=99)

    def fit_and_score(sever: bool) -> float:
        torch.manual_seed(1)
        model = build_model(spec, n_class=2)
        if sever:
            merge = model._merge
            model._merge = lambda up, skips: merge(
                up, [torch.zeros_like(s) for s in skips]
            )
        overfit(model, train_x, train_y, steps=150, lr=5e-3)
        return dice(model, test_x, test_y)

    with_skips = fit_and_score(sever=False)
    without_skips = fit_and_score(sever=True)

    assert with_skips > without_skips + 0.15, (
        f"severing the skips barely mattered ({with_skips:.4f} vs {without_skips:.4f}); "
        "either the decoder is not reading them, or the task does not need them"
    )
