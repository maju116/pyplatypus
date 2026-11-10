"""Which device a run uses, and what it tells somebody about it.

Two answers to one question that must not be able to disagree. Until `0.8.0a38` the
chooser asked about CUDA and nothing else, and the report lived in the R package's shim -
so a Python user whose GPU the build could not use was told nothing, and an Apple Silicon
Mac trained on its processor with a working backend in the same wheel.
"""

from __future__ import annotations

import pytest

import pyplatypus
from pyplatypus.device import (
    BACKENDS,
    available_backend,
    device_report,
    pick_device,
)


def test_a_requested_device_wins_over_anything_available():
    """A run that says `cpu` means it, whatever the machine has."""
    assert pick_device("cpu").type == "cpu"


def test_the_chooser_and_the_report_name_the_same_backend():
    """The pair this module exists to keep together: a report saying an accelerator is
    present beside a chooser that ignores it is worse than either mistake alone."""
    report = device_report()
    chosen = pick_device().type

    if report["backend"] is None:
        assert chosen == "cpu"
    else:
        assert chosen == report["backend"]


def test_every_backend_asked_about_is_one_torch_could_have(monkeypatch):
    """The list is asked backend by backend, so a name torch does not carry must answer
    False rather than raising - which is what happens on every machine for two of three."""
    import torch

    for name in BACKENDS:
        backend = getattr(torch, name, None)
        assert backend is None or hasattr(backend, "is_available")


def test_a_backend_that_raises_while_answering_is_not_available(monkeypatch):
    """`is_available()` can raise - a build with no CUDA support at all has done it - and
    a chooser that propagates that turns a usable processor into a failed run."""
    import torch

    class Exploding:
        @staticmethod
        def is_available():
            raise RuntimeError("no driver here")

    monkeypatch.setattr(torch, "cuda", Exploding, raising=False)
    monkeypatch.setattr(torch, "mps", Exploding, raising=False)
    monkeypatch.setattr(torch, "xpu", Exploding, raising=False)

    assert available_backend() is None
    assert pick_device().type == "cpu"


def test_the_report_is_plain_data_for_the_trip_into_r():
    """R reads this across the bridge, so every value has to survive the crossing: strings,
    booleans, a list of strings, and nothing else."""
    report = device_report()

    assert isinstance(report["torch"], str)
    assert isinstance(report["cuda_build"], str)
    assert isinstance(report["cuda_available"], bool)
    assert report["backend"] is None or isinstance(report["backend"], str)
    assert isinstance(report["device"], str)
    assert all(isinstance(arch, str) for arch in report["arch_list"])


def test_an_unusable_card_is_reported_only_when_there_is_no_backend(monkeypatch):
    """`gpu_present_but_unusable` is the one field worth acting on, and it is absent when
    an accelerator is working - there is nothing to warn about."""
    report = device_report()

    if report["backend"] is None:
        assert "gpu_present_but_unusable" in report
    else:
        assert "gpu_present_but_unusable" not in report


def test_the_report_is_reachable_from_the_package_root():
    """It is what R calls and what a Python user is told to run, so it is not buried in a
    submodule the way `pick_device` was."""
    assert pyplatypus.device_report()["torch"] == device_report()["torch"]


# ------------------------------------------------- a machine this one is not
# Everything above asks the real machine, and this one has CUDA - so the bug this module
# was written for is invisible to it: restricting the backends to CUDA alone changes no
# answer here, and neither does a report that hardcodes "cuda". These two fake the machine
# instead, which is the only way to ask the question on the hardware we have. No count
# here: this comment said "these two" while there were four below it, which is a number in
# prose with nothing holding it.
class _Says:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available

    @staticmethod
    def get_device_name(index):  # pragma: no cover - only reached when cuda is faked on
        return "a card"


def _machine(monkeypatch, *, cuda=False, mps=False, xpu=False):
    import torch

    monkeypatch.setattr(torch, "cuda", _Says(cuda), raising=False)
    monkeypatch.setattr(torch, "mps", _Says(mps), raising=False)
    monkeypatch.setattr(torch, "xpu", _Says(xpu), raising=False)


def test_an_apple_silicon_machine_uses_its_accelerator(monkeypatch):
    """Every Mac sold since 2020, and this package trained on their processors.

    `version.cuda` is forced to `None` because that is what a macOS wheel has, and without
    it this test described a machine that cannot exist. It is also what it took for the
    last assertion to be deterministic on every runner rather than a reading of whichever
    torch the matrix installed.
    """
    import torch

    _machine(monkeypatch, mps=True)
    monkeypatch.setattr(torch.version, "cuda", None)

    assert available_backend() == "mps"
    assert pick_device().type == "mps"
    assert device_report()["backend"] == "mps"
    assert device_report()["device"] == "mps"
    # The assertion this test went without, with the contradictory dictionary in its hands
    # the whole time: it called `device_report()` twice and never asked what the report
    # said about the build. `#468`, found on an M3 Pro rather than here.
    assert device_report()["cuda_build"] == "none"


def test_an_intel_gpu_is_used_when_it_is_the_only_one(monkeypatch):
    _machine(monkeypatch, xpu=True)

    assert available_backend() == "xpu"
    assert pick_device().type == "xpu"


def test_cuda_is_preferred_when_several_answer(monkeypatch):
    """Every published set of weights here was trained on CUDA, so a number from another
    backend is not obviously comparable - the order is a decision, not an accident."""
    _machine(monkeypatch, cuda=True, mps=True, xpu=True)

    assert available_backend() == "cuda"


def test_a_machine_with_nothing_says_so(monkeypatch):
    _machine(monkeypatch)

    assert available_backend() is None
    assert pick_device().type == "cpu"
    assert device_report()["backend"] is None


@pytest.mark.parametrize("accelerator", [None, "mps", "xpu"])
def test_an_accelerated_machine_is_never_called_a_processor(monkeypatch, accelerator):
    """`#468`, measured on an M3 Pro on 2026-11-10 - the first time this path ran anywhere.

    A macOS wheel has no CUDA and a working `mps` backend, and the report used to answer
    `cuda_build` with `"cpu-only build"` on it: three consecutive lines of R's
    `platypus_device()`, the first denying the two under it, on the machine the fit was
    about to run on. The `None` row is the control - there the processor IS the answer, so
    the scan would be wrong to object, and what it holds is that the value does not depend
    on the backend.
    """
    import torch

    _machine(monkeypatch, **({accelerator: True} if accelerator else {}))
    monkeypatch.setattr(torch.version, "cuda", None)

    report = device_report()

    assert report["cuda_build"] == "none"
    if accelerator is None:
        return
    assert report["backend"] == accelerator
    claims = {
        key: value
        for key, value in report.items()
        if isinstance(value, str) and "cpu" in value.lower()
    }
    assert not claims, f"an {accelerator} machine is called a processor by {claims}"


def test_a_cuda_build_is_named_by_its_version_whatever_runs_the_work(monkeypatch):
    """The other side of the same property, and the reason `mps` is the backend here: the
    value tracks `torch.version.cuda` and nothing else. A build compiled for CUDA on a
    machine using another accelerator is an ordinary state - this one's own `pascal` extra
    reaches it - and the version is still the honest answer to what the build carries."""
    import torch

    _machine(monkeypatch, mps=True)
    monkeypatch.setattr(torch.version, "cuda", "12.6")

    assert device_report()["cuda_build"] == "12.6"
