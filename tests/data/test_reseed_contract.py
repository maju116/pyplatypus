"""The contract the loader's worker hook asks for by name, held over every builder.

`reseed_augmentation_in_worker` finds its method with `getattr(augmenter, "reseed", None)`
and returns silently when it is absent. That is not a contract a reader can rely on and it
is not one a type checker sees: **no type checker runs in this repository**, so a `Protocol`
declaring `reseed` is a sentence rather than a guarantee.

`pyplatypus#351` is what the gap cost. `AlbumentationsBoxAugmenter` never had the method,
the hook was attached to `make_detection_loader` for two releases, every detection worker
drew one sequence, and the whole suite stayed green - 2731 tests. `0.8.0a60` added the
method and declared it on both protocols; `pyplatypus#355` is this file, because a
declaration nothing reads is the shape `_brand.yml` was in for weeks.

What is asserted here is the contract rather than either implementation:

* every builder a specification can reach returns something carrying a callable `reseed`;
* **and that reseeding does something**, because a method whose body is `pass` satisfies
  the first half and is exactly the mutation `#351`'s batch caught;
* and that `reseed` is a *seed* rather than a randomiser - the same seed gives the same
  next draw - because a hook seeding workers from the clock would make a run
  irreproducible while passing everything above.

The builders are enumerated off the module rather than listed, so a third one added for a
future task fails this file instead of quietly inheriting nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyplatypus.data import augmentation as module
from pyplatypus.data.augmentation import build_augmenter, build_box_augmenter
from pyplatypus.spec.components import AugmentationStep

SIZE = 64


def rotation() -> list[AugmentationStep]:
    """One transform that always fires and moves pixels rather than their values."""
    return [AugmentationStep(name="Affine", params={"rotate": [-40, 40], "p": 1.0})]


def picture() -> np.ndarray:
    image = np.zeros((SIZE, SIZE, 3), np.uint8)
    image[10:30, 8:24] = 255
    return image


def image_draw(augmenter) -> bytes:
    out, _ = augmenter(picture(), None)
    return np.asarray(out).tobytes()


def box_draw(augmenter) -> bytes:
    boxes = np.array([[8, 10, 24, 30]], dtype=np.float32)
    out, _, _ = augmenter(picture(), boxes, np.array([0], dtype=np.int64))
    return np.asarray(out).tobytes()


BUILDERS = {
    "build_augmenter": (
        lambda seed: build_augmenter(rotation(), rank=2, seed=seed),
        image_draw,
    ),
    "build_box_augmenter": (
        lambda seed: build_box_augmenter(rotation(), input_shape=(SIZE, SIZE), seed=seed),
        box_draw,
    ),
}


def test_these_are_all_the_builders_there_are():
    """The control that makes this file a guard rather than two spot checks.

    A third builder - a future task's augmenter - would otherwise inherit none of the
    assertions below and fail nothing, which is `#351` word for word.
    """
    found = {
        name
        for name in dir(module)
        if name.startswith("build_") and "augmenter" in name and not name.startswith("_")
    }
    assert found == set(BUILDERS), (
        "the module's augmenter builders and the ones this file holds have diverged: "
        f"{sorted(found)} against {sorted(BUILDERS)}"
    )


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_augmenter_carries_the_method_the_worker_hook_asks_for(name):
    builder, _ = BUILDERS[name]
    augmenter = builder(1)
    assert callable(getattr(augmenter, "reseed", None)), (
        f"{name} returns an augmenter with no `reseed`, so "
        "`reseed_augmentation_in_worker` is a no-op for it and every worker draws the "
        "same sequence - `pyplatypus#351`"
    )


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_reseeding_changes_the_next_draw(name):
    """The half a `pass` body satisfies, and the reason the test above is not enough.

    **COMPARED AT THE SAME POSITION, which the first version of this test was not.** It
    reseeded one augmenter to 7, drew, reseeded it to 8 and drew again - and a pipeline's
    generator advances with every call, so those two draws differ whether or not `reseed`
    does anything. Measured: with the box augmenter's `reseed` body replaced by `pass`, that
    version reported **8 passed** - a check that could not fail on the one mutation this file
    exists to catch, in the file written to close exactly that gap.

    Two fresh augmenters built from one construction seed, each reseeded once and drawn
    once, are the same position in both arms. A `pass` body then leaves both at the
    construction seed's first draw and they come out equal.
    """
    builder, draw = BUILDERS[name]

    seven = builder(1)
    seven.reseed(7)
    eight = builder(1)
    eight.reseed(8)

    assert draw(seven) != draw(eight), (
        f"{name}: two different seeds gave one draw at the same position, so `reseed` did nothing"
    )


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_reseeding_is_a_seed_rather_than_a_randomiser(name):
    """A hook seeding workers from the clock passes everything above and breaks a run.

    `#341` is the promise: a run repeats. The workers have to differ from each other **and**
    the run as a whole has to repeat, which only holds if the same seed starts the same
    sequence.
    """
    builder, draw = BUILDERS[name]

    first = builder(1)
    first.reseed(7)
    second = builder(1)
    second.reseed(7)

    assert draw(first) == draw(second), f"{name}: one seed gave two different draws"


def test_the_check_can_fail():
    """The negative control: an augmenter without the method is reported.

    Without this the assertion above is a check nobody has watched fail, which §4 is a list
    of - and this one costs two lines.
    """

    class WithoutReseed:
        def __call__(self, image, mask=None):  # pragma: no cover - never called
            return image, mask

    assert not callable(getattr(WithoutReseed(), "reseed", None))
