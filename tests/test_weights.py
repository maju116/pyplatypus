"""Naming weights, and refusing the ones that do not belong.

Nothing here touches the network. The Hub path is exercised by replacing the one function that
downloads - if these tests needed huggingface.co they would fail on an air-gapped machine and
in a CI run that hit a rate limit, and a test that fails for reasons unrelated to the code is
worse than no test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from pyplatypus import from_dict
from pyplatypus.models import build_model
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.weights import (
    Published,
    WeightsError,
    describe,
    export_weights,
    known_weights,
    load_into,
    resolve_weights,
)


def model_spec(**overrides) -> SegmentationModel:
    block = {"name": "m", "input_shape": (32, 32), "channels": 3, "n_class": 2,
             "blocks": 2, "filters": 4}
    block.update(overrides)
    return SegmentationModel(**block)


@pytest.fixture
def trained(tmp_path):
    """A model and its weights on disk, exported the way publishing would."""
    spec = model_spec()
    model = build_model(spec)
    path = export_weights(model, spec, tmp_path / "run.safetensors",
                          extra={"data": "synthetic", "licence": "MIT"})
    return spec, model, path


# ----------------------------------------------------------------- exporting
def test_export_writes_safetensors_and_a_sidecar(trained):
    _, _, path = trained
    assert path.name == "run.safetensors"
    assert path.exists()

    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["architecture"] == "u_net"
    assert sidecar["input_shape"] == [32, 32]
    assert sidecar["n_class"] == 2
    assert sidecar["rank"] == 2
    assert sidecar["parameters"] > 0
    # Whatever the caller adds rides along: provenance belongs with the file, not in a memory.
    assert sidecar["data"] == "synthetic"
    assert sidecar["licence"] == "MIT"


def test_the_extension_is_added_if_it_is_missing(tmp_path):
    spec = model_spec()
    path = export_weights(build_model(spec), spec, tmp_path / "weights")
    assert path.suffix == ".safetensors"


def test_exported_weights_load_back_into_the_same_model(trained):
    spec, model, path = trained
    fresh = build_model(spec)

    before = {name: value.clone() for name, value in fresh.state_dict().items()}
    load_into(fresh, str(path), spec)

    # Something actually changed, and it matches the model that was exported.
    changed = any(not torch.equal(before[name], value)
                  for name, value in fresh.state_dict().items())
    assert changed
    for name, value in model.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[name])


def test_a_round_trip_survives_a_prediction(trained):
    """Not only the numbers: the loaded model has to produce the same output."""
    spec, model, path = trained
    fresh = build_model(spec)
    load_into(fresh, str(path), spec)

    batch = torch.rand(1, 3, 32, 32)
    model.eval()
    fresh.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch), fresh(batch), atol=1e-6)


# ------------------------------------------------------------------ refusing
def test_weights_for_a_different_class_count_are_refused(trained):
    """The case that matters most: a mismatch the shapes would catch anyway is easy, but the
    sidecar is what catches the ones they would not."""
    _, _, path = trained
    other = model_spec(n_class=4)
    with pytest.raises(WeightsError, match="n_class"):
        load_into(build_model(other), str(path), other)


def test_weights_for_a_different_architecture_are_refused(trained):
    _, _, path = trained
    other = model_spec(architecture="linknet")
    with pytest.raises(WeightsError, match="architecture"):
        load_into(build_model(other), str(path), other)


def test_weights_for_a_different_width_are_refused(trained):
    _, _, path = trained
    other = model_spec(filters=8)
    with pytest.raises(WeightsError, match="filters"):
        load_into(build_model(other), str(path), other)


def test_every_disagreement_is_listed_at_once(trained):
    """One run should say everything that is wrong, not the first thing."""
    _, _, path = trained
    other = model_spec(n_class=3, filters=16)
    with pytest.raises(WeightsError) as raised:
        load_into(build_model(other), str(path), other)
    message = str(raised.value)
    assert "n_class" in message and "filters" in message


def test_weights_without_a_sidecar_still_load_if_they_fit(tmp_path):
    """Someone else's weights need not follow our convention. They are checked by torch
    instead, which catches a shape mismatch and nothing subtler - and that is worth saying
    rather than refusing the file."""
    spec = model_spec()
    model = build_model(spec)
    path = export_weights(model, spec, tmp_path / "bare.safetensors")
    path.with_suffix(".json").unlink()

    fresh = build_model(spec)
    assert load_into(fresh, str(path), spec) is None


def test_weights_that_do_not_fit_say_so_in_plain_words(tmp_path):
    spec = model_spec()
    path = export_weights(build_model(spec), spec, tmp_path / "bare.safetensors")
    path.with_suffix(".json").unlink()          # no sidecar, so torch is the only check

    other = model_spec(filters=16)
    with pytest.raises(WeightsError, match="does not fit this model"):
        load_into(build_model(other), str(path), other)


# ------------------------------------------------------------------ naming
def test_a_local_path_is_accepted(trained):
    _, _, path = trained
    assert resolve_weights(str(path)) == path


def test_a_missing_local_file_mentions_the_hub_form(tmp_path):
    with pytest.raises(WeightsError, match="hf://owner/repo"):
        resolve_weights(str(tmp_path / "absent.safetensors"))


def test_an_unknown_registry_name_lists_what_is_known():
    with pytest.raises(WeightsError, match="no published weights called"):
        resolve_weights("nothing-like-this")


def test_a_hub_reference_without_a_commit_is_refused():
    """The rule the whole module is built on: a name means one set of numbers forever."""
    with pytest.raises(WeightsError, match="names no commit"):
        resolve_weights("hf://maju116/platypus-weights/dsbowl-unet.safetensors")


def test_main_is_not_accepted_as_a_version():
    with pytest.raises(WeightsError, match="`main` is not a version"):
        resolve_weights("hf://maju116/platypus-weights/dsbowl-unet.safetensors")


def test_a_malformed_hub_reference_is_refused():
    with pytest.raises(WeightsError, match="not a Hub reference"):
        resolve_weights("hf://only-an-owner@abc123")


def test_the_registry_listing_is_readable():
    listing = known_weights()
    assert isinstance(listing, dict)
    for name, description in listing.items():
        assert isinstance(name, str) and isinstance(description, str)


# ------------------------------------------------- the Hub path, without the Hub
@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Stand in for `hf_hub_download`, so the Hub path is tested offline.

    Skips when `huggingface_hub` is genuinely absent, because it is an optional dependency and a
    plain install should be able to run the suite. CI installs it on purpose - otherwise these
    would skip forever and the Hub path would ship untested behind a green run.

    Returns the directory acting as the repo, so a test can put files in it and assert on what
    was asked for.
    """
    pytest.importorskip("huggingface_hub")
    asked = []

    def download(repo_id, filename, revision=None, **kwargs):
        asked.append((repo_id, filename, revision))
        served = tmp_path / "hub" / filename
        if not served.exists():
            from huggingface_hub.errors import EntryNotFoundError

            raise EntryNotFoundError(f"{filename} not in {repo_id}")
        return str(served)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    (tmp_path / "hub").mkdir(parents=True, exist_ok=True)
    return tmp_path / "hub", asked


def test_a_hub_reference_downloads_the_file_and_its_sidecar(fake_hub, tmp_path):
    served, asked = fake_hub
    spec = model_spec()
    export_weights(build_model(spec), spec, served / "dsbowl-unet.safetensors",
                   extra={"data": "BBBC038v1"})

    path = resolve_weights("hf://maju116/platypus-weights/dsbowl-unet.safetensors@a1b2c3d")
    assert Path(path).exists()
    assert asked[0] == ("maju116/platypus-weights", "dsbowl-unet.safetensors", "a1b2c3d")
    # The sidecar is fetched at the same revision, and found by `describe`.
    assert asked[1] == ("maju116/platypus-weights", "dsbowl-unet.json", "a1b2c3d")
    assert describe(Path(path))["data"] == "BBBC038v1"


def test_a_repo_without_a_sidecar_is_not_an_error(fake_hub):
    served, _ = fake_hub
    spec = model_spec()
    export_weights(build_model(spec), spec, served / "bare.safetensors")
    (served / "bare.json").unlink()

    path = resolve_weights("hf://someone/else/bare.safetensors@deadbee")
    assert Path(path).exists()


def test_a_registry_name_resolves_through_the_hub(fake_hub, monkeypatch):
    """What a user will actually write. The entry carries the commit, so the name cannot drift."""
    served, asked = fake_hub
    spec = model_spec()
    export_weights(build_model(spec), spec, served / "dsbowl-unet.safetensors")

    monkeypatch.setitem(
        __import__("pyplatypus.weights", fromlist=["REGISTRY"]).REGISTRY,
        "dsbowl-unet",
        Published(repo="maju116/platypus-weights", filename="dsbowl-unet.safetensors",
                  revision="a1b2c3d4e5f6", description="U-Net on the 2018 Data Science Bowl"),
    )

    path = resolve_weights("dsbowl-unet")
    assert Path(path).exists()
    assert asked[0][2] == "a1b2c3d4e5f6"


def test_a_failed_download_says_what_it_was_looking_for(fake_hub):
    with pytest.raises(WeightsError, match="could not fetch"):
        resolve_weights("hf://maju116/platypus-weights/absent.safetensors@a1b2c3d")


# ------------------------------------------------------------ through the spec
def test_a_spec_can_name_local_weights_and_skip_training(tmp_path, nested_root):
    """`weights` with `fit: false` is the whole point of publishing them: load and predict,
    without training anything."""
    from pyplatypus import Engine

    spec = model_spec(input_shape=(32, 32), channels=3)
    path = export_weights(build_model(spec), spec, tmp_path / "published.safetensors")

    engine = Engine(from_dict({
        "data": {"train_path": str(nested_root), "validation_path": str(nested_root),
                 "colormap": [[0, 0, 0], [255, 255, 255]]},
        "models": [{"name": "m", "input_shape": [32, 32], "channels": 3, "n_class": 2,
                    "blocks": 2, "filters": 4, "batch_size": 2,
                    "weights": str(path), "fit": False}],
    }), device="cpu")
    engine.fit()

    assert engine.runs["m"].trained is False
    assert engine.predict("m", split="validation").shape[1:] == (32, 32, 2)


def test_the_engine_exports_what_it_trained(tmp_path, nested_root):
    from pyplatypus import Engine

    engine = Engine(from_dict({
        "data": {"train_path": str(nested_root), "validation_path": str(nested_root),
                 "colormap": [[0, 0, 0], [255, 255, 255]]},
        "models": [{"name": "m", "input_shape": [32, 32], "channels": 3, "n_class": 2,
                    "blocks": 2, "filters": 4, "batch_size": 2, "epochs": 1}],
    }), device="cpu")
    engine.fit()

    path = engine.export_weights("m", tmp_path / "trained", data="the nested fixture")
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["data"] == "the nested fixture"
    assert sidecar["input_shape"] == [32, 32]


def test_exporting_a_model_that_was_not_trained_lists_the_ones_there_are(nested_root, tmp_path):
    from pyplatypus import Engine
    from pyplatypus.engine import EngineError

    engine = Engine(from_dict({
        "data": {"train_path": str(nested_root), "validation_path": str(nested_root),
                 "colormap": [[0, 0, 0], [255, 255, 255]]},
        "models": [{"name": "m", "input_shape": [32, 32], "channels": 3, "n_class": 2,
                    "blocks": 2, "filters": 4, "batch_size": 2, "epochs": 1}],
    }), device="cpu")
    engine.fit()
    with pytest.raises(EngineError, match="no model called"):
        engine.export_weights("nope", tmp_path / "x")


def test_the_sidecar_counts_parameters_the_way_the_engine_does(trained):
    """A state dict holds buffers as well as parameters - BatchNorm's running statistics - so
    summing it is a bigger number than the parameter count the comparison table prints. Both are
    real, so both are recorded, under names that say which is which."""
    _, model, path = trained
    sidecar = json.loads(path.with_suffix(".json").read_text())

    assert sidecar["parameters"] == sum(v.numel() for v in model.parameters())
    assert sidecar["state_dict_numel"] >= sidecar["parameters"]


def test_asking_for_a_name_without_huggingface_hub_says_which_command_fixes_it(monkeypatch):
    """The whole point of the dependency being optional is that its absence is survivable, and
    that is a promise nothing was testing. Simulated by hiding the module from imports, which is
    what a plain `pip install pyplatypus` looks like.
    """
    import builtins

    real_import = builtins.__import__

    def without_hub(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("hidden for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_hub)

    with pytest.raises(WeightsError) as raised:
        resolve_weights("hf://maju116/platypus-weights/dsbowl-unet.safetensors@a1b2c3d")

    message = str(raised.value)
    assert "pyplatypus[hub]" in message
    # And it offers the way out that needs no network at all.
    assert "give the path" in message


def test_a_local_file_still_works_without_huggingface_hub(monkeypatch, trained):
    """Which is the claim that matters for an air-gapped install: no Hub, no problem."""
    import builtins

    _, model, path = trained
    real_import = builtins.__import__

    def without_hub(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("hidden for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_hub)

    spec = model_spec()
    fresh = build_model(spec)
    load_into(fresh, str(path), spec)
    for name, value in model.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[name])


# ------------------------------------------------------------- the registry itself
def test_the_published_entries_are_pinned_to_a_full_commit():
    """Checked offline, because CI must not depend on huggingface.co. What is verifiable without
    the network is the promise the registry makes: every entry names a full 40-character commit
    rather than a branch or a tag, either of which can be moved to different numbers later."""
    from pyplatypus.weights import REGISTRY

    assert REGISTRY, "the registry is empty; a published name should be listed here"
    for name, entry in REGISTRY.items():
        assert re.fullmatch(r"[0-9a-f]{40}", entry.revision), f"{name} is not pinned"
        assert entry.filename.endswith(".safetensors"), f"{name} is not safetensors"
        assert "/" in entry.repo, f"{name} has no owner in its repo id"
        assert entry.reference.startswith("hf://")


def test_the_dsbowl_entry_says_what_it_cannot_do():
    """The description is what someone reads before using it, so the limitation belongs in it:
    these weights do not separate touching nuclei, and a count taken from them would be wrong."""
    from pyplatypus.weights import REGISTRY

    description = REGISTRY["dsbowl-unet"].description
    assert "BBBC038" in description
    assert "instance" in description.lower()


def test_a_shape_mismatch_is_rendered_as_a_tuple(tmp_path):
    """The exact wording of this line is in a released vignette.

    The fingerprint moved onto the spec and started comparing `input_shape` as a list,
    because that is what JSON returns - which quietly turned `(256, 256)` into
    `[256, 256]` in the message, and the vignette into a document showing output the
    package no longer produces. Nothing else would have noticed.
    """
    from pyplatypus.models import build_model
    from pyplatypus.spec.models import SegmentationModel
    from pyplatypus.weights import WeightsError, export_weights, load_into

    trained = SegmentationModel(name="a", input_shape=(256, 256))
    written = export_weights(build_model(trained), trained,
                             tmp_path / "w.safetensors")

    other = SegmentationModel(name="b", input_shape=(160, 160))
    with pytest.raises(WeightsError) as caught:
        load_into(build_model(other), str(written), other)
    assert "input_shape: weights say (256, 256), the model says (160, 160)" in str(
        caught.value
    )
