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


@pytest.fixture
def tiny_root_for_workers(tmp_path):
    """Four **identical** images, so a difference between them is the augmentation.

    Separate from `tiny_root` on purpose: that fixture varies the images by index, which is
    exactly what a test about two workers drawing the same sequence must not do.
    """
    for split in ("train", "valid"):
        for n in range(4):
            sample = tmp_path / split / f"s{n}"
            (sample / "images").mkdir(parents=True)
            (sample / "masks").mkdir(parents=True)
            image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
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


def test_a_detection_run_that_augments_is_reproducible_too(detection_config):
    """The construction seed of the BOX pipeline, which nothing held. `pyplatypus#362`.

    `#341` repaired augmentation seeding and gave segmentation
    `test_the_same_seed_gives_the_same_run_when_the_model_augments`. The test below is
    detection's twin and **configures no augmentation**, which is `#341`'s own sentence -
    *"none of its configurations augmented anything"* - one task over. The two worker tests
    further down run at `num_workers = 2`, where the hook reseeds each worker from
    `torch.initial_seed()`, so they cannot see the construction seed either.

    Measured: removing `seed=seed` from `AlbumentationsBoxAugmenter`'s `Compose` left the
    whole suite green - 2748 passed - while removing it from the image augmenter's failed.
    Two fits of this configuration then gave train_loss 79.00900 and 45.98642 where the
    shipped code gives 58.24984 twice.

    `num_workers` is left at the engine's default of 0 on purpose: that is where the
    construction seed is the only thing in play, and it is what an R user gets unless they
    ask otherwise.
    """
    from pyplatypus import build_engine

    detection_config["models"][0]["epochs"] = 1
    detection_config["models"][0]["augmentation"] = [
        {"name": "Affine", "params": {"rotate": [-40, 40], "p": 1.0}}
    ]

    def once(seed):
        config = {**detection_config, "seed": seed}
        return build_engine(from_dict(config), device="cpu").fit()["d"].records[-1]["train_loss"]

    assert once(1) == once(1), "two augmented detection runs at one seed differ"

    # The control: an assertion of equality passes on a pipeline that does nothing at all,
    # so a different seed has to give a different answer.
    assert once(1) != once(2), "the seed changed nothing, so this test is asserting nothing"


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


def _four_hashes(root, *, workers: int, seed: int = 3):
    """The images one epoch serves, hashed, through `workers` loader processes.

    Four identical files and a rotation that always fires, so every difference between the
    four outputs is the augmentation and nothing else.
    """
    import hashlib

    from pyplatypus import Engine
    from pyplatypus.training.torch_data import make_loader

    config = {
        "seed": seed,
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
                "epochs": 1,
                "batch_size": 1,
                "augmentation": [{"name": "Rotate", "params": {"limit": 90, "p": 1.0}}],
            }
        ],
    }
    engine = Engine(from_dict(config), device="cpu", check_masks=False)
    model = engine.spec.models[0]
    base = engine.dataset(model, "train", augmented=True)
    loader = make_loader(base, batch_size=1, num_workers=workers)
    return [hashlib.sha1(images.numpy().tobytes()).hexdigest()[:8] for images, _ in loader]


def test_two_loader_workers_do_not_draw_the_same_augmentations(tiny_root_for_workers):
    """`reseed_augmentation_in_worker`, which nothing held until this test.

    A worker is forked with a copy of the augmentation pipeline's generator **state**, so
    without a `worker_init_fn` every worker draws the same sequence - "reproducible, and
    wrong in a way that quietly halves the variety of a run with four workers", as that
    function's own docstring says. The whole suite passed with `worker_init_fn` set to
    `None` in all three loaders, and with `reseed()`'s body replaced by `pass`: the four
    tests above all run at `num_workers = 0`, where a worker function is never called.

    Measured, four identical images at `num_workers = 2`: as shipped the four outputs are
    all different, and with the hook removed they arrive in two identical pairs -
    `a11023b3, a11023b3, afc146b6, afc146b6` - because the loader hands alternating indices
    to the two workers and each draws from the same state. `pyplatypus#348`.

    The single-worker reading is the control that the rotation is what varies them: one
    process draws four times from one advancing generator, so four different outputs there
    say nothing about workers, and the pair pattern above is what distinguishes them.
    """
    one = _four_hashes(tiny_root_for_workers, workers=0)
    assert len(set(one)) == 4, "the pipeline did not vary within one process"

    two = _four_hashes(tiny_root_for_workers, workers=2)
    assert len(set(two)) == 4, f"two workers drew the same sequence: {two}"


def test_two_worker_runs_at_one_seed_still_agree(tiny_root_for_workers):
    """The other half, and the reason the test above is not satisfied by randomising.

    Each worker's sequence is derived from `torch.initial_seed()`, which is itself derived
    from the run's seed - so the workers differ from each other and the run as a whole
    still repeats. A hook seeding the workers from the clock would pass the test above and
    fail this one.
    """
    assert _four_hashes(tiny_root_for_workers, workers=2) == _four_hashes(
        tiny_root_for_workers, workers=2
    )


@pytest.fixture
def detection_root_for_workers(tmp_path, voc_sample):
    """Four **identical** annotated images, for the detector's half of the same question.

    `detection_root` varies its boxes by index on purpose - anchors need shapes to fit -
    which is exactly what a test about two workers drawing one sequence must not do.
    """
    for split in ("train", "valid"):
        for n in range(4):
            voc_sample(tmp_path / split, f"s{n}", [(10, 10, 50, 50), (96, 20, 130, 34)], [0, 1])
    return tmp_path


def _four_detection_hashes(root, detection_classes, *, workers: int, seed: int = 3):
    """The detector's `_four_hashes`: one epoch's images, hashed, through `workers` processes."""
    import hashlib

    from pyplatypus import build_engine
    from pyplatypus.training.torch_data import make_detection_loader

    config = {
        "task": "object_detection",
        "seed": seed,
        "data": {
            "train_path": str(root / "train"),
            "validation_path": str(root / "valid"),
            "classes": list(detection_classes),
        },
        "models": [
            {
                "name": "d",
                "input_shape": [128, 128],
                "epochs": 1,
                "batch_size": 1,
                "anchors_per_grid": 2,
                "optimizer": {"name": "adam", "learning_rate": 1e-3},
                # Always fires, and it moves the pixels rather than their values: a
                # transform with `p < 1` would make an unaugmented sample look like a
                # worker drawing the same thing as its neighbour.
                "augmentation": [{"name": "Affine", "params": {"rotate": [-40, 40], "p": 1}}],
            }
        ],
    }
    engine = build_engine(from_dict(config), device="cpu")
    base = engine.dataset(engine.spec.models[0], "train", augmented=True)
    loader = make_detection_loader(base, batch_size=1, num_workers=workers)
    # Three things since `pyplatypus#359`: the image's boxes travel beside the targets,
    # because the ignore mask cannot be built from what the target could hold.
    return [hashlib.sha1(images.numpy().tobytes()).hexdigest()[:8] for images, _, _ in loader]


def test_two_detection_workers_do_not_draw_the_same_augmentations(
    detection_root_for_workers, detection_classes
):
    """The half of `pyplatypus#341`'s worker fix the detector was left out of.

    `reseed_augmentation_in_worker` asks the augmenter for a `reseed` by name and returns
    silently when there is none. `AlbumentationsBoxAugmenter` had none, so the hook was
    attached to `make_detection_loader` and did nothing for every release from `0.8.0a58`
    to `0.8.0a59`, and **no test in the suite could fail on it**: removing the hook from
    this loader left all 2731 tests green. `pyplatypus#351`.

    Measured on four identical images at `num_workers = 2`: before the method existed the
    four outputs arrived as two identical pairs - `39af39b6, 39af39b6, fabb8142, fabb8142` -
    because the loader hands alternating indices to the two workers and each drew from the
    same generator state. The zero-worker reading is the control that the rotation is what
    varies them at all.
    """
    one = _four_detection_hashes(detection_root_for_workers, detection_classes, workers=0)
    assert len(set(one)) == 4, "the box pipeline did not vary within one process"

    two = _four_detection_hashes(detection_root_for_workers, detection_classes, workers=2)
    assert len(set(two)) == 4, f"two detection workers drew the same sequence: {two}"


def test_two_detection_worker_runs_at_one_seed_still_agree(
    detection_root_for_workers, detection_classes
):
    """And the detector repeats, which is what stops the test above being satisfied by a clock."""
    assert _four_detection_hashes(
        detection_root_for_workers, detection_classes, workers=2
    ) == _four_detection_hashes(detection_root_for_workers, detection_classes, workers=2)


def test_a_reseed_keeps_the_boxes_on_the_object(detection_classes):
    """A reseed must not leave the box processor behind, which is not free to assume.

    The box pipeline is built with `bbox_params` inside a `warnings.catch_warnings()`, so
    a reseed that reset the image transform's generator and not the processor's state
    would move the picture and leave the boxes where they were - a silent mislabelling
    rather than a crash. Asserted by giving one object a frame to sit in and demanding that
    the returned box still contains the pixels that moved.
    """
    from pyplatypus.data.augmentation import build_box_augmenter
    from pyplatypus.spec.components import AugmentationStep

    augmenter = build_box_augmenter(
        [AugmentationStep(name="HorizontalFlip", params={"p": 1.0})],
        input_shape=(64, 64),
        seed=1,
    )
    image = np.zeros((64, 64, 3), np.uint8)
    image[10:20, 4:14] = 255  # one bright square near the left edge
    boxes = np.array([[4, 10, 14, 20]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)

    augmenter.reseed(99)
    out_image, out_boxes, out_labels = augmenter(image, boxes, labels)

    x0, y0, x1, y1 = (round(float(v)) for v in out_boxes[0])
    inside = out_image[y0:y1, x0:x1]
    assert inside.max() == 255, "the box no longer covers the object after a reseed"
    # The control: a flip did happen, so this is not passing because nothing moved.
    assert x0 >= 32, f"HorizontalFlip did not move the box, it is still at {out_boxes[0]}"
    assert list(out_labels) == [0]


@pytest.fixture
def labelled_root_for_workers(tmp_path):
    """One directory per class, four **identical** pictures in each."""
    picture = np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)
    for name in ("benign", "malignant"):
        where = tmp_path / name
        where.mkdir(parents=True)
        for n in range(4):
            Image.fromarray(picture).save(where / f"{n:04d}.png")
    return tmp_path


def _classification_hashes(root, *, workers: int, seed: int = 3):
    """The classifier's `_four_hashes`. Eight identical pictures, one rotation, batch of one."""
    import hashlib

    from pyplatypus.classification_engine import ClassificationEngine
    from pyplatypus.training.torch_data import make_classification_loader

    config = {
        "task": "classification",
        "seed": seed,
        "data": {
            "train_path": str(root),
            "classes": ["benign", "malignant"],
            "split": {"fractions": [0.75, 0.25], "group_by": None, "seed": 0},
        },
        "models": [
            {
                "name": "c",
                "input_shape": [16, 16],
                "blocks": 2,
                "filters": 4,
                "epochs": 1,
                "batch_size": 1,
                "optimizer": {"name": "adam", "learning_rate": 0.01},
                "augmentation": [{"name": "Rotate", "params": {"limit": 90, "p": 1.0}}],
            }
        ],
    }
    engine = ClassificationEngine(from_dict(config))
    base = engine.dataset("c", "train", augment=True)
    loader = make_classification_loader(base, batch_size=1, num_workers=workers)
    return [hashlib.sha1(images.numpy().tobytes()).hexdigest()[:8] for images, _ in loader]


def test_two_classification_workers_do_not_draw_the_same_augmentations(
    labelled_root_for_workers,
):
    """The third loader carrying the hook, and the third one nothing held.

    `AlbumentationsAugmenter` does have `reseed`, so this path was never broken - and it
    was never asserted either: removing `worker_init_fn` from `make_classification_loader`
    left the whole suite green, exactly as it did for the detector. A working mechanism with
    no test is one release away from the detector's case. `pyplatypus#351`.
    """
    one = _classification_hashes(labelled_root_for_workers, workers=0)
    assert len(set(one)) == len(one), "the pipeline did not vary within one process"

    two = _classification_hashes(labelled_root_for_workers, workers=2)
    assert len(set(two)) == len(two), f"two classification workers drew the same sequence: {two}"
