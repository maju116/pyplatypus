"""`deterministic`, the opt-in that makes a seed mean something on a card.

`seed` is honest about where it stops: measured on a GTX 1070, two runs of one
configuration at one seed agreed to twelve decimals on the processor and differed in the
fourth after one epoch on CUDA, because cuDNN picks algorithms nothing constrained. This is
the half that constrains them, and it is an opt-in because the cost arrives at an unusual
time - an operation with no deterministic implementation raises **when it is reached**,
which no other refusal in this package does.

**What is asserted here and what cannot be.** That the flag reaches torch, that it is not
reached when nobody asked, that the environment variable cuBLAS needs is set without
trampling an existing one, that the specification refuses the half-promise, and that a real
run still trains with it on. The thing the field is *for* - two CUDA runs agreeing - needs a
card, and CI has none; `tests/test_seeding.py` says the same about the other half rather than
asserting something it cannot see. platypus#122.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from pyplatypus import Engine, from_dict
from pyplatypus.errors import ConfigError
from pyplatypus.training.trainer import seed_everything


@pytest.fixture
def restore_torch():
    """Leave the process as it was found.

    `torch.use_deterministic_algorithms(True)` is global and sticky, so a test that sets it
    for real would hand every later test in the session an interpreter where some operations
    raise. The monkeypatched tests below do not touch it; this is for the one that does.
    """
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    yield
    torch.use_deterministic_algorithms(before[0])
    torch.backends.cudnn.deterministic = before[1]
    torch.backends.cudnn.benchmark = before[2]


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


def config(root, **top):
    return {
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
                "batch_size": 2,
            }
        ],
        **top,
    }


def record_calls(monkeypatch):
    """Watch what `seed_everything` asks of torch, without letting it ask."""
    asked = {}
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", lambda value, **kw: asked.update(algorithms=value)
    )
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False, raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True, raising=False)
    return asked


def test_the_flag_reaches_torch(monkeypatch):
    asked = record_calls(monkeypatch)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    seed_everything(7, deterministic=True)

    assert asked == {"algorithms": True}
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_nothing_is_asked_when_nobody_asked(monkeypatch):
    """The point of an opt-in. If this passed anyway, the cost would be paid by every run
    that only wanted a seed - and paid as a refusal mid-training on some of them."""
    asked = record_calls(monkeypatch)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    seed_everything(7)

    assert asked == {}
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    assert "CUBLAS_WORKSPACE_CONFIG" not in __import__("os").environ


def test_the_cublas_variable_is_set_but_never_overwritten(monkeypatch):
    """cuBLAS needs it from CUDA 10.2 on, and torch raises without it. An existing value is
    somebody's deliberate choice - `:16:8` trades memory for speed - so it stands."""
    import os

    record_calls(monkeypatch)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    seed_everything(7, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    seed_everything(7, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_a_specification_refuses_the_half_promise(tiny_root):
    """Constraining the algorithms while the draw still varies is slower and no more
    repeatable, so it is refused where every other configuration problem is - before
    anything runs."""
    with pytest.raises(ConfigError) as raised:
        from_dict(config(tiny_root, deterministic=True), check_paths=False)
    message = str(raised.value)
    assert "deterministic" in message
    assert "seed" in message


def test_a_seed_and_the_flag_together_are_accepted(tiny_root):
    spec = from_dict(config(tiny_root, seed=1, deterministic=True), check_paths=False)
    assert spec.deterministic is True
    assert spec.seed == 1


def test_the_default_is_off_and_says_so_in_the_schema():
    from pyplatypus import spec_schema

    field = spec_schema()["$defs"]["SegmentationSpec"]["properties"]["deterministic"]
    assert field["default"] is False
    # The cost is the reason this is not the default, so a reader has to meet it here.
    assert "raises" in field["description"]
    assert "seed" in field["description"]


def test_a_run_with_it_on_still_trains(tiny_root, restore_torch):
    """The flag's own failure mode is an operation refusing mid-run, so the useful check is
    that this package's ordinary path does not contain one. On the processor, where CI lives,
    it also has to give the same numbers twice - that is already true without the flag, which
    is why it is not the claim being made."""
    engine = Engine(
        from_dict(config(tiny_root, seed=5, deterministic=True)), device="cpu", check_masks=False
    )
    history = engine.fit()["u"]
    assert np.isfinite(history.records[-1]["train_loss"])
    assert torch.are_deterministic_algorithms_enabled()
