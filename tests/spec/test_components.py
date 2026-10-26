"""Losses, optimisers and callbacks: the parts that used to be typed `Any`."""

import inspect
import re

import pytest

from pyplatypus import ConfigError, available_transforms, from_dict

CLASS_LOSSES = [
    "iou",
    "dice",
    "cce",
    "cce_dice",
    "focal",
    "tversky",
    "focal_tversky",
    "combo",
    "lovasz",
]
#: Named on a `separation: flows` block rather than on a model, which is the only place a
#: continuous target exists - a model's own `loss` scores the classes at every segmentation
#: task and refuses these by name (`pyplatypus#223`). This list used to be reached through a
#: semantic configuration, which the engine accepted and ran.
CONTINUOUS_LOSSES = ["mse", "l1", "smooth_l1"]


@pytest.mark.parametrize("name", CLASS_LOSSES)
def test_every_class_loss_is_reachable(config, name):
    config["models"][0]["loss"] = {"name": name}
    assert from_dict(config).models[0].loss.name == name


@pytest.mark.parametrize("name", CONTINUOUS_LOSSES)
def test_every_continuous_loss_is_reachable_where_it_belongs(name):
    """Reachable, and reachable from the one place it belongs. Both halves are the test:
    the flow block accepts it, and the model that owns the block does not."""
    model = {"name": "m", "input_shape": [64, 64], "separation": {"name": "flows"}}
    spec = from_dict(
        {
            "task": "instance_segmentation",
            "data": {"train_path": ".", "validation_path": "."},
            "models": [{**model, "separation": {"name": "flows", "loss": {"name": name}}}],
        },
        check_paths=False,
    )
    assert spec.models[0].separation.loss.name == name

    with pytest.raises(ConfigError, match="class channels"):
        from_dict(
            {
                "task": "instance_segmentation",
                "data": {"train_path": ".", "validation_path": "."},
                "models": [{**model, "loss": {"name": name}}],
            },
            check_paths=False,
        )


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


# ---------------------------------------------------- one hand on the learning rate
# A schedule recomputes the rate from the optimiser's initial value every epoch, so two of
# them fight and the list order decides; a schedule beside `reduce_lr_on_plateau` erases
# what the plateau cut. Measured: with both, the rate follows the cosine exactly - 0.0976,
# 0.0905, 0.0794 from 0.1 - where the plateau alone reaches 1e-6 over the same six epochs.
def test_two_schedules_are_refused(config):
    config["models"][0]["callbacks"] = [
        {"name": "cosine_annealing"},
        {"name": "step_decay", "every": 10},
    ]
    with pytest.raises(ConfigError, match="two learning rate schedules"):
        from_dict(config)


def test_a_schedule_beside_the_plateau_callback_is_refused(config):
    config["models"][0]["callbacks"] = [
        {"name": "linear_annealing"},
        {"name": "reduce_lr_on_plateau"},
    ]
    with pytest.raises(ConfigError, match="erases what the plateau callback cut"):
        from_dict(config)


def test_a_schedule_beside_swa_is_allowed(config):
    """They are not in conflict: `swa` pins a constant tail and the order is declared."""
    config["models"][0]["callbacks"] = [
        {"name": "step_decay", "milestones": [1]},
        {"name": "swa", "start": 0.75},
    ]
    assert len(from_dict(config).models[0].callbacks) == 2


def test_step_decay_needs_exactly_one_way_of_saying_when(config):
    for callback in (
        {"name": "step_decay"},
        {"name": "step_decay", "every": 10, "milestones": [10]},
    ):
        config["models"][0]["callbacks"] = [callback]
        with pytest.raises(ConfigError, match="exactly one of"):
            from_dict(config)


def test_step_decay_refuses_milestones_that_are_not_increasing(config):
    config["models"][0]["callbacks"] = [{"name": "step_decay", "milestones": [30, 10]}]
    with pytest.raises(ConfigError, match="strictly increasing"):
        from_dict(config)


def test_step_decay_refuses_a_factor_that_raises_the_rate(config):
    config["models"][0]["callbacks"] = [{"name": "step_decay", "factor": 1.5, "every": 10}]
    with pytest.raises(ConfigError):
        from_dict(config)


# ------------------------------------------- when a callback begins, and what it keeps
def test_a_warm_up_cannot_be_said_two_ways(config):
    for callback in (
        {"name": "early_stopping", "start": 0.1, "start_epoch": 5},
        {"name": "swa", "start": 0.5, "start_epoch": 5},
    ):
        config["models"][0]["callbacks"] = [callback]
        with pytest.raises(ConfigError, match="not both"):
            from_dict(config)


def test_swa_still_takes_either_on_its_own(config):
    """`start` carries a default, so the refusal has to read what was written rather than
    what the value is - 0.75 is both the default and a legitimate thing to ask for."""
    config["models"][0]["callbacks"] = [{"name": "swa", "start_epoch": 5}]
    assert from_dict(config).models[0].callbacks[0].start_epoch == 5

    config["models"][0]["callbacks"] = [{"name": "swa", "start": 0.5}]
    assert from_dict(config).models[0].callbacks[0].start == pytest.approx(0.5)


def test_a_condition_on_a_quantity_this_model_will_never_report_is_refused(config):
    config["models"][0]["callbacks"] = [
        {"name": "model_checkpoint", "path": "out.pt", "when": {"val_dice": 0.9}}
    ]
    with pytest.raises(ConfigError, match="condition on 'val_dice'"):
        from_dict(config)


def test_a_condition_on_a_quantity_it_does_report_is_accepted(config):
    config["models"][0]["metrics"] = [{"name": "dice"}]
    config["models"][0]["callbacks"] = [
        {"name": "model_checkpoint", "path": "out.pt", "when": {"val_dice": 0.9}}
    ]
    assert from_dict(config).models[0].callbacks[0].when == {"val_dice": 0.9}


def test_a_budget_has_to_be_a_positive_number_of_hours(config):
    config["models"][0]["callbacks"] = [{"name": "stop_after", "hours": 0}]
    with pytest.raises(ConfigError):
        from_dict(config)


def test_two_weight_averagers_are_refused(config):
    """Both replace the model when training ends, so the second averages what the first
    already wrote - in whichever order the list happened to have them."""
    config["models"][0]["callbacks"] = [{"name": "ema"}, {"name": "swa"}]
    with pytest.raises(ConfigError, match="two weight averagers"):
        from_dict(config)


def test_one_averager_is_fine_either_way(config):
    for callback in ({"name": "ema"}, {"name": "swa"}):
        config["models"][0]["callbacks"] = [callback]
        assert len(from_dict(config).models[0].callbacks) == 1


def test_a_decay_outside_zero_to_one_is_refused(config):
    for decay in (0, 1, 1.5, -0.1):
        config["models"][0]["callbacks"] = [{"name": "ema", "decay": decay}]
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

    @staticmethod
    def _stated_inventory():
        """The albumentations version the docstring's inventory was measured on, and its total.

        Read out of the prose rather than written down here, because a second copy of the
        version is the drift this whole file is about. The docstring says "albumentations 2.0.8
        offers 118 transforms", so it is already honest about being version-scoped - what was
        missing is the tests noticing.

        Why that matters: `0.8.0a45`'s `floors` job runs the suite at the oldest versions the
        manifest permits, and the inventory there is a different inventory - 113 transforms at
        albumentations 1.4.23, 114 at 2.0.0, 118 at 2.0.8. A test that asserts 118 without
        asking which version it is looking at would make the floor `==2.0.8`, which is a pin
        and not a floor.
        """
        albumentations = pytest.importorskip("albumentations")
        doc = inspect.getdoc(available_transforms) or ""
        # `[^.]*` would not do here: the sentence names the albumentations version, so the
        # span before the count contains "2.0.8".
        stated = re.search(r"albumentations\s+(\S+)\s+offers\s+(\d+)\s+transforms", doc, re.DOTALL)
        assert stated, "the docstring no longer states its version and total in a readable form"
        return stated.group(1), int(stated.group(2)), albumentations.__version__

    @staticmethod
    def _as_tuple(version: str) -> tuple[int, ...]:
        """`"1.4.23"` as `(1, 4, 23)`, for comparing against the version the prose names.

        Parsed here rather than with `packaging`, which is not a dependency of this package
        and would have to be either added or skipped around - and a skip in the gate that
        decides whether to assert is a gate nobody can read.
        """
        return tuple(int(part) for part in re.findall(r"\d+", version)[:3])

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
        albumentations'. That job was Windows with Python 3.13, while Windows with 3.10 - the
        matrix had a 3.10 cell until 0.8.0a44, and this was measured 2026-10-11 - and
        macOS with 3.13 both answered 87, so neither axis explains it. There is no portable
        number there to pin, and the docstring says so rather than claiming one.

        This docstring said "eleven of the thirty-one" while naming nine and four, which is
        thirteen, and the real figure was seventeen once the categories were counted instead
        of recalled. A sum nothing adds up is the cheapest kind of wrong number to ship, and
        `test_no_refusal_is_unexplained` is the part that now holds it.
        """
        version, total, installed = self._stated_inventory()
        measured = len(available_transforms())

        if installed != version:
            # The subject is absent rather than wrong: this count is a property of one
            # albumentations release and the docstring says which. What is still assertable
            # is that the prose is version-scoped at all, which is what makes it checkable
            # wherever that version IS installed - and that the listing works here.
            assert measured > 0, (
                f"albumentations {installed} is installed and offers no transforms at all"
            )
            return

        assert total == measured, f"the docstring says {total} transforms; measured {measured}"

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
        """The mutation, built in: without the large probe, `Crop` goes back to absent.

        Without this the test above would pass for any reason at all - an albumentations
        release that changed those defaults would make it green while the escalation did
        nothing.

        **`Crop` rather than all four, and the first version of this test asserting all four
        was right to fail.** `ubuntu-latest` with Python 3.10 - 2026-10-11, while the matrix
        still had that cell - reported a three-name difference,
        because `Superpixels` at 8x8 is **not deterministic**: measured here, 4 of 60 attempts
        succeed, since it samples `n_segments` and whether the segmentation divides by zero
        depends on the draw. So which names the escalation adds is a random variable at the
        small size, and the only members of that set worth asserting are the ones that refuse
        8x8 every time. `Crop` is the clearest - its default crop box is 1024x1024 - and the
        two directions that *are* portable are asserted alongside it: escalating only ever
        adds, and it never adds anything outside the four.

        That randomness is also the answer to a question this file used to leave open. The
        87-against-88 disagreement was attributed to "Windows with Python 3.13" with no
        mechanism claimed. There was no platform axis: eight consecutive runs in one process
        on one machine gave 87, 87, 87, 87, 87, 87, 88, 87.
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

        added = set(with_escalation) - set(small_only)

        # The two portable directions, asserted on every version: escalating only ever adds,
        # and `<` is strict so it has to add SOMETHING - which is the mutation control the
        # docstring above is about. Neither depends on which transform it is.
        assert small_only < with_escalation, "escalating must only ever add names"
        assert not added - set(self.SIZE_DEPENDENT), (
            f"the escalation added something unaccounted for: {sorted(added - set(self.SIZE_DEPENDENT))}"
        )

        # And the named member, which is not portable and is the stronger claim where it
        # applies. `Crop`'s default crop box is 1024x1024 in the version the docstring was
        # measured on; at albumentations 2.0.0 the escalation adds the other three and not
        # `Crop`, so asserting the name everywhere would be asserting that release's
        # defaults. `0.8.0a45`'s floors job is what found this.
        version, _, installed = self._stated_inventory()
        if installed != version:
            return
        assert "Crop" in added, (
            "`Crop` refuses an 8x8 probe on every platform and every run, so the escalation "
            f"is what lists it; the escalation added {sorted(added)}"
        )

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

        # OLDER than the table was measured on and the categories are not this test's to
        # hold: at albumentations 1.4.23 `RandomFog` raises `AxisError: axis 2 is out of
        # bounds`, and at 2.0.0 `LongestMaxSize` raises `Unsupported number of dimensions` -
        # transforms that simply had no volume support yet, each announcing it in its own
        # way. Naming those messages one by one is how a closed set stops being closed, and
        # the closed set is this test's entire value.
        #
        # **The comparison is `<` and not `!=`, and that asymmetry is the design.** A count
        # belongs to exactly one release, so `test_the_docstring_total_is_the_measured_one`
        # gates on `!=`. A taxonomy is expected to hold FORWARD - an upgrade may move the
        # numbers and is not expected to invent a new kind of refusal, which is what this
        # test's own docstring promises - so anything at or above the stated version is
        # asserted, and a newer albumentations inventing a fifth category still fails here.
        # Gating this one on `!=` would have silenced it at the next lock bump, which is the
        # one moment it exists for.
        version, _, installed = self._stated_inventory()
        if self._as_tuple(installed) < self._as_tuple(version):
            return

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
