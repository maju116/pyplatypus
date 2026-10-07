"""Training must be quiet.

A warning emitted once per batch is a warning emitted tens of thousands of times per
run, which trains people to ignore the log. Caught when a fresh install pulled a newer
torch than the one used during development.
"""

import warnings

from pyplatypus.data import SegmentationDataset, discover
from pyplatypus.models import build_model
from pyplatypus.spec.components import DiceMetric
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.training import Trainer, make_loader


def test_a_training_epoch_emits_no_warnings(nested_root, binary_data):
    spec = SegmentationModel(
        name="m",
        input_shape=(32, 32),
        channels=3,
        blocks=2,
        filters=4,
        batch_size=2,
        epochs=1,
        metrics=[DiceMetric()],
    )
    loader = make_loader(
        SegmentationDataset(discover(nested_root, binary_data).samples, spec, binary_data),
        batch_size=2,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Trainer(build_model(spec, n_class=2), spec, device="cpu").fit(loader, loader)

    noisy = [str(w.message) for w in caught if "requires_grad" in str(w.message)]
    assert noisy == [], f"training emitted {len(noisy)} gradient warnings: {noisy[:2]}"
