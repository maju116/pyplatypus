"""Pretrained encoders: the seam, the refusals, and what is actually pretrained.

Nothing here downloads weights, and until `#294` this file claimed that the one question
needing a network was exercised by a test that is skipped without one. **There was no such
test**, and the claim had split a question that is really two:

* whether ImageNet weights *arrive* genuinely needs a network, and is still not tested here;
* whether they **survive construction** needs nothing at all, and is
  `test_a_transferred_backbone_survives_construction` below.

The second is what `#294` was: `initialise` walked every convolution under the network and
overwrote the backbone immediately after timm had loaded it. The test below would have
failed the day that shipped, offline, in milliseconds.
"""

import builtins
import pathlib

import pytest
import torch

from pyplatypus import engine as engine_module
from pyplatypus.models.encoders import PretrainedEncoder, verify_encoder
from pyplatypus.models.layers import ModelError
from pyplatypus.models.unet import build_model
from pyplatypus.spec.models import SegmentationModel

timm = pytest.importorskip("timm", reason="the `encoders` extra is not installed")


def spec(**overrides):
    base = {"name": "m", "input_shape": (64, 64), "channels": 3, "blocks": 4, "filters": 16}
    return SegmentationModel(**{**base, **overrides})


def encoder(**overrides):
    base = {"in_channels": 3, "blocks": 4, "filters": 16}
    return PretrainedEncoder(overrides.pop("name", "resnet34"), **{**base, **overrides})


# --- the contract ------------------------------------------------------------------------


def test_a_backbone_satisfies_the_encoder_contract():
    """The check that catches a half-resolution encoder has to pass for the real thing."""
    built = encoder()
    verify_encoder(built, 2, 3, 5)


def test_level_zero_is_ours_and_at_full_resolution():
    """An ImageNet stem strides by 2, so the finest level cannot come from the backbone.
    It is one block of ours, and the resolution is what the decoder's head needs."""
    built = encoder()
    features = built(torch.zeros(1, 3, 64, 64))
    assert built.own_level_zero
    assert built.stem is not None
    assert features[0].shape[-2:] == (64, 64)
    assert [f.shape[-1] for f in features] == [64, 32, 16, 8, 4]


def test_a_backbone_that_offers_full_resolution_is_used_directly():
    """vgg16 reports a 1/1 level, so prepending one of ours would waste it."""
    built = encoder(name="vgg16")
    assert not built.own_level_zero
    assert built.stem is None
    assert built(torch.zeros(1, 3, 64, 64))[0].shape[-2:] == (64, 64)


def test_a_whole_model_builds_and_keeps_the_input_size():
    model = build_model(spec(), encoder=encoder(), n_class=2)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)


@pytest.mark.parametrize("architecture", ["u_net", "u_net_plus_plus", "res_u_net", "linknet"])
def test_every_architecture_accepts_a_backbone(architecture):
    """The backbone replaces the contracting path; the decoder is still the spec's choice."""
    model = build_model(spec(architecture=architecture), encoder=encoder(), n_class=2)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)


def test_gradients_reach_our_stage_as_well_as_the_backbone():
    model = build_model(spec(), encoder=encoder(), n_class=2)
    model(torch.randn(1, 3, 64, 64)).sum().backward()
    unused = [name for name, p in model.named_parameters() if p.grad is None]
    assert unused == []


@pytest.mark.parametrize("blocks", [2, 3, 4, 5])
def test_blocks_selects_how_many_backbone_stages_to_use(blocks):
    built = encoder(blocks=blocks)
    assert len(built.channels) == blocks + 1
    verify_encoder(built, 2, 3, blocks + 1)


def test_a_transferred_backbone_survives_construction():
    """What arrived pretrained is still there after the model is built. `#294`.

    `pretrained=False` on purpose: the question is whether construction *keeps* whatever
    the backbone was given, and random weights answer it exactly as well as ImageNet ones
    while needing no network. Measured when this was written: without the fix, 15 of 15
    convolutions in `resnet18` moved and one tensor's norm went 2.418 to 11.618.

    The second half is the control, and it is the one that makes this a test rather than an
    assertion that `initialise` does nothing: the full-resolution stage inside
    `PretrainedEncoder` is **ours**, starts random, and must still be initialised.
    """
    built = encoder(name="resnet18")
    before = {name: p.detach().clone() for name, p in built.named_parameters() if p.dim() == 4}
    assert any(name.startswith("backbone") for name in before), "no backbone convolutions found"
    assert any(name.startswith("stem") for name in before), "no stage of our own found"

    build_model(spec(encoder="resnet18"), n_class=2, encoder=built)

    after = {name: p.detach() for name, p in built.named_parameters() if p.dim() == 4}
    moved = {name for name in before if not torch.equal(before[name], after[name])}
    theirs = {name for name in moved if name.startswith("backbone")}
    assert not theirs, f"construction overwrote {len(theirs)} transferred convolutions"
    assert {name for name in moved if name.startswith("stem")}, (
        "our own stage was not initialised, so this test would pass on a model that "
        "initialises nothing at all"
    )


def test_what_the_trainer_protects_is_exactly_what_construction_left_alone():
    """One answer to *what arrived pretrained*, paired rather than asserted twice.

    `Trainer` reads `model.encoder.transferred` for the freeze and for
    `encoder_learning_rate`; `UShapedNet` reads the same property to decide what not to
    initialise. The two could drift if either side grew its own idea - a name pattern, a
    list of module types - so this compares the weights construction did not touch against
    the parameters that property offers, by name.
    """
    built = encoder()
    before = {name: p.detach().clone() for name, p in built.named_parameters() if p.dim() == 4}
    model = build_model(spec(encoder="resnet34"), n_class=2, encoder=built)
    after = {name: p.detach() for name, p in built.named_parameters() if p.dim() == 4}

    untouched = {name for name in before if torch.equal(before[name], after[name])}
    protected = {
        f"backbone.{name}"
        for name, p in model.encoder.transferred.named_parameters()
        if p.dim() == 4
    }
    assert protected, "the property offered nothing; the comparison would be two empty sets"
    assert untouched == protected


# --- the refusals -----------------------------------------------------------------------


def test_a_volume_is_refused_by_the_spec_before_anything_is_built():
    """Caught while the configuration is read, so a 3D run fails before a download."""
    with pytest.raises(ValueError, match="ImageNet is images"):
        spec(input_shape=(64, 64, 32), encoder="resnet34")


def test_the_encoder_itself_also_refuses_rank_three():
    """Belt and braces: the spec is the usual route in, but the class is public."""
    with pytest.raises(ModelError, match="2D only"):
        encoder(rank=3)


def test_pretrained_without_an_encoder_is_refused():
    """Otherwise the flag reads as transfer learning and does nothing at all."""
    with pytest.raises(ValueError, match="needs `encoder`"):
        spec(pretrained=True)


def test_a_patch_based_backbone_is_refused_by_name_and_reason():
    """convnext and swin start at 1/4. The missing 1/2 level is where a U-shaped decoder
    recovers fine detail, and inventing it would carry no pretrained information at all."""
    with pytest.raises(ModelError) as raised:
        encoder(name="convnext_tiny")
    message = str(raised.value)
    assert "convnext_tiny" in message
    assert "1/2" in message
    assert "resnet" in message  # and it says what to use instead


def test_asking_for_more_blocks_than_the_backbone_has_is_refused():
    with pytest.raises(ModelError, match=r"needs \[2, 4, 8, 16, 32, 64\]"):
        encoder(blocks=6)


def test_an_unknown_backbone_name_is_refused_with_somewhere_to_look():
    with pytest.raises(ModelError) as raised:
        encoder(name="not_a_real_backbone_xyz")
    assert "timm.list_models" in str(raised.value)


# --- channels ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_channel_counts_other_than_three(channels):
    """Grayscale CT has one, BraTS has four; ImageNet has three and timm adapts the stem."""
    built = PretrainedEncoder("resnet34", in_channels=channels, blocks=4, filters=16)
    verify_encoder(built, 2, channels, 5)


# --- normalisation ----------------------------------------------------------------------


def test_input_statistics_are_applied_only_with_the_weights_that_expect_them():
    """ImageNet statistics on a from-scratch run are meaningless numbers."""
    assert not encoder().normalises


def test_the_absence_of_timm_names_the_command_that_fixes_it():
    """The extra being optional is a promise that its absence is survivable and says so."""
    real_import = builtins.__import__

    def without_timm(name, *args, **kwargs):
        if name == "timm" or name.startswith("timm."):
            raise ImportError("hidden for this test")
        return real_import(name, *args, **kwargs)

    original = builtins.__import__
    builtins.__import__ = without_timm
    try:
        with pytest.raises(ModelError) as raised:
            encoder()
    finally:
        builtins.__import__ = original

    message = str(raised.value)
    assert "pyplatypus[encoders]" in message
    # And it names the route that needs nothing extra and works at both ranks.
    assert "built-in encoder" in message


# --- the delivery, rather than the two ends -----------------------------------------------
#
# `pyplatypus#385`. `#381`'s tests held `EncoderModel.normalises_input`, which computes the
# fingerprint, and `PretrainedEncoder(normalise=)`, constructed **directly** - and nothing
# held the line between them. Both forwards could be deleted with the whole suite green, and
# the result is the silence `#370` was written to refuse: a fingerprint recording a
# normalisation the network does not apply, every key agreeing, predictions made on
# un-normalised input.
#
# So these ask the question of the function a run calls. `_encoder_for` is never otherwise
# executed by this suite - coverage marks its `return PretrainedEncoder(` as missed - which
# is why all thirteen of its arguments were unheld, `pretrained=` included.


@pytest.mark.parametrize(
    ("asked", "expected"),
    [
        ({"pretrained": False, "normalise": True}, True),
        ({"pretrained": False, "normalise": None}, False),
        ({"pretrained": False, "normalise": False}, False),
    ],
)
def test_a_segmentation_run_s_own_encoder_route_carries_normalise(asked, expected):
    """The engine's `_encoder_for`, which is the line `build_model` is handed."""
    from pyplatypus.engine import _encoder_for

    model_spec = spec(encoder="resnet34", **asked)
    built = _encoder_for(model_spec)
    assert built is not None
    assert built.normalises is expected
    # And the fingerprint agrees with what was built, which is the pair that matters: the
    # defect is not either value, it is the two disagreeing.
    assert model_spec.weights_fingerprint()["input_normalised"] is expected


@pytest.mark.parametrize(
    ("asked", "expected"),
    [
        ({"pretrained": False, "normalise": True}, True),
        ({"pretrained": False, "normalise": None}, False),
    ],
)
def test_a_classification_run_s_own_encoder_route_carries_normalise(asked, expected):
    """The same question of the other builder, which resolves its encoder itself."""
    from pyplatypus.models.classifier import _encoder_for
    from pyplatypus.spec.classification import ClassificationModel

    model_spec = ClassificationModel(
        name="c", input_shape=(64, 64), channels=3, encoder="resnet34", **asked
    )
    built = _encoder_for(model_spec)
    assert built.normalises is expected
    assert model_spec.weights_fingerprint()["input_normalised"] is expected


# --- pretrained, and the one route that decides whether a backbone is transferred at all --
#
# `maju116/pyplatypus#395`. The tenth pass said *"all thirteen arguments of that call are
# unheld here, `pretrained=` included"*; `0.8.0a66` closed one of the thirteen and the
# eleventh pass measured what the rest cost. `pretrained` is the one with a consequence: with
# the forward gone, `SegmentationModel(encoder="resnet18", pretrained=True)` reports
# `weights_fingerprint()["input_normalised"] == True` while the backbone is randomly
# initialised and **no download is attempted**. The fingerprint records a transfer that did
# not happen.
#
# A spy rather than a download: the question is whether the flag arrives, and timm's own
# behaviour on arrival is timm's.


@pytest.mark.parametrize("asked", [True, False])
@pytest.mark.parametrize("route", ["segmentation", "classification"])
def test_pretrained_arrives_at_timm_through_both_encoder_routes(monkeypatch, route, asked):
    """Spied at `timm.create_model`, which is the only place the flag can mean anything."""
    import timm

    seen = []
    real = timm.create_model

    def spy(name, *args, **kwargs):
        # `verify_encoder` probes with `pretrained=False` before the real build, so only
        # the call that builds the backbone is recorded - the one carrying `in_chans`.
        if "in_chans" in kwargs:
            seen.append(kwargs.get("pretrained"))
        return real(name, *args, **{**kwargs, "pretrained": False})

    # `encoders` imports timm lazily inside `_import_timm()`, so the spy goes on the module
    # that function returns rather than on an attribute of `encoders`.
    monkeypatch.setattr(timm, "create_model", spy)

    if route == "segmentation":
        from pyplatypus.engine import _encoder_for

        built = _encoder_for(spec(encoder="resnet18", pretrained=asked, blocks=2, filters=8))
    else:
        from pyplatypus.models.classifier import _encoder_for
        from pyplatypus.spec.classification import ClassificationModel

        built = _encoder_for(
            ClassificationModel(
                name="c", input_shape=(64, 64), channels=3, encoder="resnet18", pretrained=asked
            )
        )

    assert built is not None
    assert seen == [asked], f"timm was asked for pretrained={seen}, the spec said {asked}"


def test_the_fingerprint_and_the_backbone_agree_about_being_transferred(monkeypatch):
    """The pair the forward exists for, which is what made this worth a test rather than a
    note: a fingerprint saying the input was normalised with ImageNet statistics, over a
    network that never loaded any."""
    import timm

    from pyplatypus.engine import _encoder_for

    real = timm.create_model
    monkeypatch.setattr(
        timm,
        "create_model",
        lambda name, *a, **k: real(name, *a, **{**k, "pretrained": False}),
    )

    asked = spec(encoder="resnet18", pretrained=True, blocks=2, filters=8)
    built = _encoder_for(asked)
    # `pretrained: true` is what makes both of these true, and the defect is either one of
    # them being true alone.
    assert asked.weights_fingerprint()["input_normalised"] is True
    assert built.normalises is True


# --- the encoder route's own thirteen arguments, enumerated --------------------------------
#
# `maju116/pyplatypus#395`. `_encoder_for` forwards thirteen fields and **no test executes
# it** except the ones written for `normalise` and `pretrained` above - which is why the
# tenth pass's sentence was *"all thirteen arguments of that call are unheld"*. The two
# builder routes are enumerated in `tests/models/test_field_delivery.py`; this is the third
# route, and the same question: does a non-default change the built encoder at all?


#: field -> (non-default, companions needed to make it reachable)
ENCODER_ROUTE_FIELDS = {
    "block_width": (3, {}),
    "activation": ("gelu", {}),
    "batch_normalization": (False, {}),
    "separable_conv": (True, {}),
    "dropout": (0.5, {}),
    "spatial_dropout": (False, {"dropout": 0.5}),
    # `#404`: this table had no completeness half, so these three could simply be absent -
    # and `blocks` and `filters` were, with the builder enumeration one directory over
    # exempting them as *shape-bearing*. They size the one stage this route owns and the
    # number of backbone stages it takes, and nothing about the output shape says so.
    "blocks": (3, {}),
    "filters": (16, {}),
    "channels": (1, {}),
}

#: The reads in `_encoder_for` that are held somewhere else, each naming where. The scan
#: below intersects with the model's own fields, so `rank` - a derived property - is not
#: here and cannot go stale as a name for something that does not exist.
NOT_ENUMERATED_HERE = {
    "encoder": "names the backbone; every row above is built with one",
    "pretrained": "spied at `timm.create_model` above - the flag, not what timm does with it",
    "normalise": "`#385`, asserted above against `weights_fingerprint()`",
}


def _encoder_fingerprint(built):
    kinds = tuple(sorted(type(module).__name__ for module in built.modules()))
    state = tuple(
        (name, tuple(t.shape), round(float(t.detach().sum()), 6))
        for name, t in sorted(built.state_dict().items())
    )
    return kinds, state


def _through_the_route(**overrides):
    from pyplatypus.engine import _encoder_for

    torch.manual_seed(0)
    # Merged rather than passed alongside, because `blocks` and `filters` are rows in the
    # table now and a row's value has to be able to replace the baseline's.
    asked = {"encoder": "resnet18", "blocks": 2, "filters": 8, **overrides}
    return _encoder_for(spec(**asked))


@pytest.mark.parametrize("field", sorted(ENCODER_ROUTE_FIELDS))
def test_a_field_reaches_the_encoder_the_route_builds(field):
    value, companions = ENCODER_ROUTE_FIELDS[field]
    default = _encoder_fingerprint(_through_the_route(**companions))
    asked = _encoder_fingerprint(_through_the_route(**companions, **{field: value}))
    assert default != asked, (
        f"`{field}={value!r}` builds the same encoder as the default through `_encoder_for`"
    )


def test_two_builds_of_one_specification_agree_through_the_route():
    """The control, as in the builder enumeration: the fingerprint reads parameter values,
    so two builds of the same spec must agree or every row above reads the draw."""
    assert _encoder_fingerprint(_through_the_route()) == _encoder_fingerprint(_through_the_route())


def _reads_in_the_route() -> set[str]:
    """Every `spec.<name>` inside `engine._encoder_for`, by AST rather than by grep.

    Scoped to that one function rather than to the file, because this is the route's own
    enumeration and a read elsewhere in `engine.py` is a different question. `#404`: the
    unit of this whole class is the read site, so the scan is pointed at a site.
    """
    import ast

    source = pathlib.Path(engine_module.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "_encoder_for":
            found = {
                child.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "spec"
            }
            assert found, "`_encoder_for` was found and reads no `spec.<field>` at all"
            return found
    raise AssertionError("no `_encoder_for` in the engine; the scan is looking in the wrong place")


def test_every_field_the_encoder_ROUTE_reads_is_enumerated_or_named():
    """What this table did not have, and the reason two reads survived behind a sentence
    about a third.

    `#404`. The builder enumeration has carried this half since it was written; this route's
    did not, so `blocks` and `filters` were missing rather than exempted - which looks
    identical from inside the file.
    """
    fields = set(SegmentationModel.model_fields)
    read = _reads_in_the_route() & fields
    assert read, "the AST scan found no field reads at all"

    covered = set(ENCODER_ROUTE_FIELDS) | set(NOT_ENUMERATED_HERE)
    missing = sorted(read - covered)
    assert not missing, (
        f"these fields are read by `_encoder_for` and neither enumerated above nor named "
        f"in NOT_ENUMERATED_HERE: {', '.join(missing)}"
    )

    stale = sorted(set(NOT_ENUMERATED_HERE) - read)
    assert not stale, f"NOT_ENUMERATED_HERE names reads `_encoder_for` does not make: {stale}"
