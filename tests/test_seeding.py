"""`seed` on the specification, which until now did nothing.

The field has been there since the first release, described as "set it if you want a
reproducible run", and no code read it. In R, `platypus_spec(seed = 1)` made a promise and
two runs of it disagreed - which is worse than no field at all, because the person who set
it stopped looking for the reason their numbers moved.

Found while wiring detection in, where the anchors are fitted by k-means and the seed had
to come from somewhere.

These tests train very small models twice, which is the only way to assert the thing that
is actually claimed. Asserting that `torch.initial_seed()` changed would pass with the
seed applied after the weights were initialised, which is exactly the mistake worth
catching.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyplatypus import Engine, from_dict


@pytest.fixture
def tiny_root(tmp_path):
    for split in ("train", "valid"):
        for n in range(3):
            sample = tmp_path / split / f"s{n}"
            (sample / "images").mkdir(parents=True)
            (sample / "masks").mkdir(parents=True)
            image = np.full((32, 32, 3), 10 * n, np.uint8)
            mask = np.zeros((32, 32, 3), np.uint8)
            mask[8:20, 8:20] = 255
            Image.fromarray(image).save(sample / "images" / "i.png")
            Image.fromarray(mask).save(sample / "masks" / "m.png")
    return tmp_path


def final_loss(root, seed):
    config = {
        "data": {"train_path": str(root / "train"),
                 "validation_path": str(root / "valid"),
                 "colormap": [[0, 0, 0], [255, 255, 255]]},
        "models": [{"name": "u", "input_shape": [32, 32], "blocks": 2, "filters": 4,
                    "epochs": 2, "batch_size": 2}],
    }
    if seed is not None:
        config["seed"] = seed
    engine = Engine(from_dict(config), device="cpu", check_masks=False)
    return engine.fit()["u"].records[-1]["train_loss"]


def test_the_same_seed_gives_the_same_run(tiny_root):
    first = final_loss(tiny_root, 7)
    second = final_loss(tiny_root, 7)
    assert first == second


def test_without_a_seed_two_runs_differ(tiny_root):
    """The other half, and the one that proves the first is not vacuous: if these agreed
    too, the test above would pass on a package that ignores `seed` entirely - which is
    the state this fixes."""
    assert final_loss(tiny_root, None) != final_loss(tiny_root, None)


def test_a_different_seed_gives_a_different_run(tiny_root):
    assert final_loss(tiny_root, 7) != final_loss(tiny_root, 8)


def test_a_detection_run_is_reproducible_too(detection_config):
    """Including the anchors, which are fitted rather than given and so are part of what a
    seed has to pin down."""
    from pyplatypus import build_engine

    detection_config["models"][0]["epochs"] = 1

    def once():
        engine = build_engine(from_dict(detection_config), device="cpu")
        history = engine.fit()["d"]
        return engine.runs["d"].anchors, history.records[-1]["train_loss"]

    first_anchors, first_loss = once()
    second_anchors, second_loss = once()
    assert first_anchors == second_anchors
    assert first_loss == second_loss
