"""Losses, optimisers and callbacks: the parts that used to be typed `Any`."""

import inspect
import re

import pytest

from pyplatypus import ConfigError, available_transforms, from_dict


@pytest.mark.parametrize(
    "name",
    [
        "iou",
        "dice",
        "cce",
        "cce_dice",
        "focal",
        "tversky",
        "focal_tversky",
        "combo",
        "lovasz",
    ],
)
def test_every_loss_is_reachable(config, name):
    config["models"][0]["loss"] = {"name": name}
    assert from_dict(config).models[0].loss.name == name


@pytest.mark.parametrize(
    "name",
    [
        "adam",
        "adamw",
        "sgd",
        "rmsprop",
        "adagrad",
        "adadelta",
        "adamax",
        "nadam",
    ],
)
def test_every_optimiser_is_reachable(config, name):
    config["models"][0]["optimizer"] = {"name": name}
    assert from_dict(config).models[0].optimizer.name == name


def test_loss_parameters_are_range_checked(config):
    config["models"][0]["loss"] = {"name": "tversky", "alpha": 1.5}
    with pytest.raises(ConfigError):
        from_dict(config)


def test_tversky_betas_complement_alpha(config):
    config["models"][0]["loss"] = {"name": "tversky", "alpha": 0.3}
    assert from_dict(config).models[0].loss.beta == pytest.approx(0.7)


def test_nesterov_without_momentum_is_refused(config):
    config["models"][0]["optimizer"] = {"name": "sgd", "nesterov": True}
    with pytest.raises(ConfigError, match="momentum"):
        from_dict(config)


def test_checkpoint_requires_a_path(config):
    config["models"][0]["callbacks"] = [{"name": "model_checkpoint"}]
    with pytest.raises(ConfigError):
        from_dict(config)


def test_not_fitting_without_weights_is_refused(config):
    config["models"][0]["fit"] = False
    with pytest.raises(ConfigError, match="weights"):
        from_dict(config)


def test_loading_weights_without_fitting_is_fine(config):
    config["models"][0]["fit"] = False
    config["models"][0]["weights"] = "dsbowl2018"
    assert from_dict(config).models[0].weights == "dsbowl2018"


def test_deep_supervision_needs_depth(config):
    config["models"][0]["blocks"] = 1
    config["models"][0]["input_shape"] = [32, 32]
    config["models"][0]["architecture"] = "u_net_plus_plus"
    config["models"][0]["deep_supervision"] = True
    with pytest.raises(ConfigError, match="at least 2 blocks"):
        from_dict(config)


def test_deep_supervision_only_makes_sense_for_the_nested_architecture(config):
    """It reads the intermediate nodes X[0][j], which only u_net_plus_plus produces."""
    config["models"][0]["deep_supervision"] = True
    with pytest.raises(ConfigError, match="u_net_plus_plus"):
        from_dict(config)


def test_deep_supervision_is_fine_on_u_net_plus_plus(config):
    config["models"][0]["architecture"] = "u_net_plus_plus"
    config["models"][0]["deep_supervision"] = True
    assert from_dict(config).models[0].deep_supervision


class TestAvailableTransforms:
    """What `available_transforms()` may and may not return.

    Until 0.7.0a2 it filtered `dir(albumentations)` by the shape of the name - capitalised,
    not starting with `Base`, `Basic` or `Dual` - which let through ten things a spec cannot
    name: eight composition classes and two parameter dataclasses. None was ever usable,
    because an `AugmentationStep` is a flat name and a dict of parameters, so naming one
    passed the check and failed when the pipeline was built.
    """

    def test_everything_returned_is_a_transform(self):
        """Type decides membership, so nothing in the list can fail to be a transform."""
        albumentations = pytest.importorskip("albumentations")
        names = available_transforms()

        assert names, "albumentations is installed, so the list cannot be empty"
        for name in sorted(names):
            member = getattr(albumentations, name)
            assert isinstance(member, type), name
            assert issubclass(member, albumentations.BasicTransform), name

    def test_the_interface_classes_are_not_offered(self):
        """The four classes albumentations exposes for subclassing, not for naming in a spec.

        They are excluded by identity because nothing in their type says what they are:
        `DualTransform()`, `ImageOnlyTransform()` and `Transform3D()` all instantiate
        without complaint. `NoOp` is the reason the module cannot decide either - a real
        transform living beside them in `albumentations.core.transforms_interface`.
        """
        names = available_transforms()

        for interface in ("BasicTransform", "DualTransform", "ImageOnlyTransform", "Transform3D"):
            assert interface not in names, interface
        assert "NoOp" in names, "a usable transform, despite sharing their module"

    def test_the_ten_that_used_to_leak_are_gone(self):
        """Named one by one, because this is the regression and a count would not show which."""
        names = available_transforms()

        composition = (
            "Compose",
            "OneOf",
            "OneOrOther",
            "RandomOrder",
            "ReplayCompose",
            "SelectiveChannelTransform",
            "Sequential",
            "SomeOf",
        )
        for name in composition + ("BboxParams", "KeypointParams"):
            assert name not in names, name

    def test_the_docstring_total_is_the_measured_one(self):
        """The prose states how many transforms there are, and this is what keeps it true.

        The defect being fixed here was never only the filter: the docstring said "71 work on
        volumes and 33 do not" long after the measurement said otherwise, because correcting a
        number nobody re-measures is not a step anyone schedules. `albumentations` is pinned
        only as `>=1.4`, so an upgrade moves this - and then the test names the new value
        rather than letting the documentation drift again.

        **Only the total is asserted, and the first version of this test asserted the volume
        count too and was right to fail.** One CI job of twelve said "the docstring says
        118/87/31; measured 118/88/30": the volume list is found by running each transform
        against a probe, and most of what it rejects fails for the probe's reasons rather than
        albumentations'. That job was Windows with Python 3.13, while Windows with 3.10 and
        macOS with 3.13 both answered 87, so neither axis explains it. There is no portable
        number there to pin, and the docstring says so rather than claiming one.

        This docstring said "eleven of the thirty-one" while naming nine and four, which is
        thirteen, and the real figure was seventeen once the categories were counted instead
        of recalled. A sum nothing adds up is the cheapest kind of wrong number to ship, and
        `test_no_refusal_is_unexplained` is the part that now holds it.
        """
        pytest.importorskip("albumentations")
        doc = inspect.getdoc(available_transforms) or ""
        # `[^.]*` would not do here: the sentence names the albumentations version, so the
        # span before the count contains "2.0.8".
        stated = re.search(r"albumentations\s+\S+\s+offers\s+(\d+)\s+transforms", doc, re.DOTALL)
        assert stated, "the docstring no longer states the total in a readable form"

        measured = len(available_transforms())
        assert int(stated.group(1)) == measured, (
            f"the docstring says {stated.group(1)} transforms; measured {measured}"
        )

    def test_the_volume_list_is_a_smaller_subset(self):
        """What is portable about rank 3: shorter, contained, and not empty.

        The count is not assertable - see above - but these three are, and together they are
        what a caller relies on: a name that passes at rank 3 is a real transform, and asking
        for volumes narrows rather than changes the answer.
        """
        pytest.importorskip("albumentations")
        images = available_transforms()
        volumes = available_transforms(rank=3)

        assert volumes, "albumentations is installed, so some transform takes a volume"
        assert volumes < images, "rank 3 must be a strict subset of rank 2"
        assert "GaussNoise" in images and "GaussNoise" not in volumes

    # The four that an 8x8 probe refused for its own reasons. Named one by one rather than
    # counted, because a count would say something moved and not which - the lesson the
    # ten-that-leaked test above is also written from.
    SIZE_DEPENDENT = ("Crop", "FrequencyMasking", "Superpixels", "TimeMasking")

    @pytest.mark.parametrize("name", SIZE_DEPENDENT)
    def test_a_transform_is_not_refused_for_the_probe_s_size(self, name):
        """Each of these takes a volume; only the small probe said otherwise.

        `Crop` is the one that shows the probe was never close: its default crop box is
        1024x1024, so no plausible small probe would have admitted it. The other three need
        32, 64 and 16 pixels, which is near enough to 8 that which of them tipped over was
        environmental - and that is the symptom the escalation removes.
        """
        pytest.importorskip("albumentations")

        assert name in available_transforms(rank=3)

    def test_the_escalation_is_what_lists_them(self):
        """The mutation, built in: with the large probe taken away, all four go back to absent.

        Without this the test above would pass for any reason at all - an albumentations
        release that changed those defaults would make it green while the escalation did
        nothing. It is also the measurement the change rests on: 87 names with one probe
        size, 91 with two, and the difference is exactly these four.
        """
        pytest.importorskip("albumentations")
        from pyplatypus.spec import components

        with_escalation = available_transforms(rank=3)

        available_transforms.cache_clear()
        original = components._PROBE_SIDES
        components._PROBE_SIDES = (8,)
        try:
            small_only = available_transforms(rank=3)
        finally:
            components._PROBE_SIDES = original
            available_transforms.cache_clear()

        assert small_only < with_escalation, "escalating must only ever add names"
        assert set(with_escalation) - set(small_only) == set(self.SIZE_DEPENDENT)

    def test_no_refusal_is_unexplained(self):
        """Every name rank 3 refuses must fail for a reason the docstring names.

        This is the part that holds the docstring's table, and it holds the *categories*
        rather than their counts, which are not portable - an albumentations upgrade is
        expected to move the numbers and is not expected to invent a new kind of refusal.
        If it does, this fails and prints the transform and the message, which is the whole
        point: the previous version of that table was a remembered sum that did not add up,
        and nothing would have noticed a fourth category appearing underneath it.
        """
        albumentations = pytest.importorskip("albumentations")
        import warnings

        import numpy as np

        from pyplatypus.spec import components

        refused = sorted(available_transforms() - available_transforms(rank=3))
        assert refused, "some transform must be refused, or this test proves nothing"

        side = max(components._PROBE_SIDES)
        volume = np.zeros((components._PROBE_DEPTH, side, side, 1), dtype=np.float32)
        mask = np.zeros((components._PROBE_DEPTH, side, side), dtype=np.uint8)

        unexplained = []
        for name in refused:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                transform = getattr(albumentations, name)(p=1)
                try:
                    albumentations.Compose([transform])(volume=volume, mask3d=mask)
                except Exception as refusal:  # noqa: BLE001 - the message is the evidence
                    message = f"{type(refusal).__name__}: {refusal}"
                else:
                    unexplained.append(f"{name}: accepted the large probe yet is not listed")
                    continue
            no_volume_support = "'images'" in message
            needs_three_channels = "3-channel" in message
            needs_a_target = "requires [" in message or "'bboxes'" in message
            if not (no_volume_support or needs_three_channels or needs_a_target):
                unexplained.append(f"{name}: {message}")

        assert not unexplained, "refusals the docstring does not account for:\n" + "\n".join(
            unexplained
        )
