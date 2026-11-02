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

**They force `device="cpu"`, and the names say so, because the claim is device-scoped.**
Measured on a GTX 1070: two runs of one configuration at one seed agreed to twelve decimals
on the processor and differed in the fourth after a single epoch on CUDA, since cuDNN picks
algorithms `seed_everything` does not constrain. The CUDA half is therefore documented on
the `seed` field and tested nowhere - a test asserting that two runs *differ* would pass by
luck and fail by luck, and CI has no card to run it on either. platypus#122.

What is repeatable on any device is asserted elsewhere and worth knowing about:
`tests/data/test_splits.py::test_the_same_seed_gives_the_same_split` holds the partition,
which is the guarantee that matters most when two results are being compared.
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


def final_loss(root, seed, augment=False):
    config = {
        "task": "semantic_segmentation",
        "data": {
            "train_path": str(root / "train"),
            "validation_path": str(root / "valid"),
            "colormap": [[0, 0, 0], [255, 255, 255]],
        },
        "models": [
            {
                "name": "u",
                "input_shape": [32, 32],
                "blocks": 2,
                "filters": 4,
                "epochs": 2,
                "batch_size": 2,
            }
        ],
    }
    if seed is not None:
        config["seed"] = seed
    if augment:
        # The configuration this suite never had, which is why it could not fail on
        # `pyplatypus#341`: the augmentation pipeline draws from albumentations' own
        # generator and nothing seeded it.
        config["models"][0]["augmentation"] = [
            {"name": "Rotate", "params": {"limit": 45, "p": 1.0}},
            {"name": "RandomBrightnessContrast", "params": {"p": 1.0}},
        ]
    engine = Engine(from_dict(config), device="cpu", check_masks=False)
    return engine.fit()["u"].records[-1]["train_loss"]


def test_the_same_seed_gives_the_same_run_on_the_processor(tiny_root):
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


def test_the_same_seed_gives_the_same_run_when_the_model_augments(tiny_root):
    """`pyplatypus#341`, and the gap this suite had: every configuration above augments
    nothing.

    `seed_everything` seeds `random`, `numpy` and `torch`. albumentations draws from its
    **own** generator, which none of those reach - so two runs at one seed differed
    whenever a model augmented, while the `seed` field's description promised "the order
    the data arrives in and its augmentation".
    """
    assert final_loss(tiny_root, 7, augment=True) == final_loss(tiny_root, 7, augment=True)


def test_without_a_seed_two_augmented_runs_differ(tiny_root):
    """The control that the test above is not vacuous: with the pipeline pinned to a
    constant rather than to the run's seed, both would pass."""
    assert final_loss(tiny_root, None, augment=True) != final_loss(tiny_root, None, augment=True)


def test_a_different_seed_augments_differently(tiny_root):
    """And the seed is what moved it, rather than the augmentation having become a no-op."""
    assert final_loss(tiny_root, 7, augment=True) != final_loss(tiny_root, 8, augment=True)


def test_a_seeded_pipeline_still_varies_from_call_to_call(tiny_root):
    """Why seeding the pipeline is the right fix, rather than a choice against variety.

    `build_box_augmenter`'s docstring used to refuse a seed on the grounds that "pinning
    the augmentation sequence would make every epoch see the same distortions". That is
    not what a seed does: it fixes where the sequence starts and the generator still
    advances. Asserted here because the sentence it replaces was an argument rather than a
    measurement, and because a fix that DID pin the sequence would pass every test above.
    """
    import numpy as np

    from pyplatypus.data.augmentation import build_augmenter
    from pyplatypus.spec.components import AugmentationStep

    steps = [
        AugmentationStep(name="Rotate", params={"limit": 45, "p": 1.0}),
        AugmentationStep(name="RandomBrightnessContrast", params={"p": 1.0}),
    ]
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)

    def three(seed):
        augmenter = build_augmenter(steps, seed=seed)
        return [augmenter(image.copy())[0].tobytes() for _ in range(3)]

    first = three(11)
    assert len(set(first)) == 3, "the seed pinned the sequence instead of its start"
    assert three(11) == first, "the same seed did not reproduce the sequence"
    assert three(12) != first, "a different seed drew the same sequence"
