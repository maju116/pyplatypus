"""Weights describe their own geometry, so a specification need not restate it.

The point is not convenience. Using somebody else's published model meant knowing its
internals - `dsbowl-unet` is four blocks of sixteen filters - and being refused for
guessing. The file has said so all along; nothing read it.
"""

import numpy as np
import pytest
from PIL import Image

from pyplatypus.engine import Engine
from pyplatypus.models.unet import build_model
from pyplatypus.spec import from_dict
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.weights import WeightsError, export_weights


@pytest.fixture
def unusual_weights(tmp_path):
    """Deliberately not the defaults: blocks=4 filters=16 would pass by coincidence."""
    spec = SegmentationModel(name="m", input_shape=(32, 32), blocks=2, filters=8)
    export_weights(build_model(spec, n_class=2), spec,
                   tmp_path / "w.safetensors", extra={"n_class": 2})
    return tmp_path / "w.safetensors"


@pytest.fixture
def tiny_root(tmp_path):
    root = tmp_path / "data"
    for i in range(2):
        sample = root / f"s{i}"
        (sample / "images").mkdir(parents=True)
        (sample / "masks").mkdir(parents=True)
        mask = np.zeros((32, 32), np.uint8)
        mask[8:20, 8:20] = 255
        Image.fromarray(mask).convert("RGB").save(sample / "images" / "a.png")
        Image.fromarray(mask).convert("RGB").save(sample / "masks" / "m.png")
    return root


def spec_for(root, weights, **model):
    return from_dict({
        "task": "semantic_segmentation",
        "data": {"train_path": str(root), "validation_path": str(root),
                 "colormap": [[0, 0, 0], [255, 255, 255]]},
        "models": [{"name": "m", "input_shape": [32, 32], "weights": str(weights),
                    "fit": False, **model}],
    })


def test_geometry_the_spec_did_not_state_comes_from_the_file(tiny_root, unusual_weights):
    engine = Engine(spec_for(tiny_root, unusual_weights), device="cpu", check_masks=False)
    engine.fit()
    # Built at the file's geometry, not at the defaults the spec would have carried.
    run = engine.runs["m"]
    assert run.spec.blocks == 2 and run.spec.filters == 8


def test_stating_it_correctly_still_works(tiny_root, unusual_weights):
    engine = Engine(spec_for(tiny_root, unusual_weights, blocks=2, filters=8),
                    device="cpu", check_masks=False)
    engine.fit()
    assert engine.runs["m"].spec.blocks == 2


def test_stating_it_wrongly_is_still_refused(tiny_root, unusual_weights):
    """Adoption loosens what may be omitted and nothing about what is checked."""
    engine = Engine(spec_for(tiny_root, unusual_weights, blocks=3),
                    device="cpu", check_masks=False)
    with pytest.raises(WeightsError, match="blocks"):
        engine.fit()


def test_input_shape_is_not_adopted(tiny_root, unusual_weights):
    """And deliberately so: the rank comes from it and is needed while the specification is
    validated, long before a sidecar can be reached without a download. A spec that cannot
    be checked offline is the air-gapped hospital problem."""
    from pyplatypus.errors import ConfigError
    with pytest.raises(ConfigError, match="input_shape"):
        from_dict({
            "task": "semantic_segmentation",
            "data": {"train_path": str(tiny_root), "validation_path": str(tiny_root),
                     "colormap": [[0, 0, 0], [255, 255, 255]]},
            "models": [{"name": "m", "weights": str(unusual_weights), "fit": False}],
        }, check_paths=False)
