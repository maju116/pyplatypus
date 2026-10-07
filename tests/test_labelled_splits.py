"""Which splits carry masks, and what that decides.

The detection engine has recorded this since detection existed. This side discovered the
same fact - it asked for the test split without masks - and then threw it away, so a test
folder of images alone built a dataset that would read masks and fail on the first item
with `MaskError: no masks to unite`, two layers below the question that was asked.

It also meant the opposite: a test split that *did* carry masks was never recognised as
scoreable, because discovery always asked for images only.
"""

import numpy as np
import pytest
from PIL import Image

from pyplatypus.engine import Engine, EngineError
from pyplatypus.errors import ConfigError
from pyplatypus.spec import from_dict

COLORMAP = [[0, 0, 0], [255, 255, 255]]


def write(root, *, masks=True, n=2, missing=()):
    for i in range(n):
        sample = root / f"s{i}"
        (sample / "images").mkdir(parents=True)
        picture = np.zeros((32, 32), np.uint8)
        picture[8:20, 8:20] = 255
        Image.fromarray(picture).convert("RGB").save(sample / "images" / "a.png")
        if masks and i not in missing:
            (sample / "masks").mkdir(parents=True)
            Image.fromarray(picture).convert("RGB").save(sample / "masks" / "m.png")
    return root


@pytest.fixture
def labelled_parts(tmp_path):
    return write(tmp_path / "train"), write(tmp_path / "valid")


def engine_for(parts, test, **kwargs):
    train, valid = parts
    return Engine(
        from_dict(
            {
                "task": "semantic_segmentation",
                "data": {
                    "train_path": str(train),
                    "validation_path": str(valid),
                    "test_path": str(test),
                    "colormap": COLORMAP,
                },
                "models": [{"name": "m", "input_shape": [32, 32]}],
            }
        ),
        device="cpu",
        check_masks=False,
        **kwargs,
    )


def test_a_test_split_with_masks_is_scoreable(labelled_parts, tmp_path):
    engine = engine_for(labelled_parts, write(tmp_path / "full"))
    assert "test" in engine.labelled
    _, mask = engine.dataset(engine.spec.models[0], "test")[0]
    assert mask is not None


def test_a_test_split_without_masks_is_not(labelled_parts, tmp_path):
    engine = engine_for(labelled_parts, write(tmp_path / "bare", masks=False))
    assert "test" not in engine.labelled
    _, mask = engine.dataset(engine.spec.models[0], "test")[0]
    assert mask is None, "the dataset should not try to read masks that are not there"


def test_scoring_an_unlabelled_split_is_refused_by_name(labelled_parts, tmp_path):
    """It used to raise `MaskError: no masks to unite` from the mask layer - true, and
    about the wrong thing. The question was "score this", and the answer is that there is
    nothing to score against."""
    engine = engine_for(labelled_parts, write(tmp_path / "bare", masks=False))
    engine.fit()
    with pytest.raises(EngineError, match="no masks"):
        engine.evaluate("test")
    with pytest.raises(EngineError, match="no masks"):
        engine.evaluate_cases("m", "test")


def test_the_refusal_says_what_does_work(labelled_parts, tmp_path):
    """A message may not leave its reader with nothing to do."""
    engine = engine_for(labelled_parts, write(tmp_path / "bare", masks=False))
    engine.fit()
    with pytest.raises(EngineError) as caught:
        engine.evaluate("test")
    assert "predict" in str(caught.value)


def test_some_masks_missing_is_an_error_not_a_demotion(labelled_parts, tmp_path):
    """The distinction taken from the detection side: falling back whenever labelled
    discovery failed would turn a split with two missing files into an unlabelled one,
    throwing away the ones that are there and reporting nothing."""
    partial = write(tmp_path / "partial", n=3, missing=(1,))
    with pytest.raises(ConfigError):
        engine_for(labelled_parts, partial)


def test_asking_for_masks_explicitly_still_works(labelled_parts, tmp_path):
    """`only_images` is still a parameter; it simply defaults to what the split has."""
    engine = engine_for(labelled_parts, write(tmp_path / "full"))
    _, mask = engine.dataset(engine.spec.models[0], "test", only_images=True)[0]
    assert mask is None
