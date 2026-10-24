"""Where the work will happen, and whether that is what anybody intended.

Two questions that must not be able to disagree: which device a run will use, and what to
tell somebody about it. `pick_device` answers the first and `device_report` the second,
and they read the same table of backends - because a report saying an accelerator is
present, beside a chooser that ignores it, is worse than either mistake alone.

**The silent failure this module exists for.** A GPU the installed build cannot use looks
exactly like no GPU at all: the run falls back to the processor, takes perhaps ten times
as long, and nothing says why. torch 2.8 and later ship CUDA 13 builds, and CUDA 13
dropped the Maxwell, Pascal and Volta generations - so a GTX 10-series card needs torch
2.7.x, which is what the `pascal` extra is. No driver update brings those cards back.
"""

from __future__ import annotations

from typing import Any

#: The accelerators asked about, in the order they are preferred. CUDA first because every
#: published set of weights here was trained on it and a number from another backend is not
#: obviously comparable; the rest in the order torch gained them.
BACKENDS = ("cuda", "mps", "xpu")


def _backend_available(name: str) -> bool:
    """Whether torch has this backend and it is usable on this machine."""
    import torch

    backend = getattr(torch, name, None)
    available = getattr(backend, "is_available", None)
    if available is None:
        return False
    try:
        return bool(available())
    except Exception:  # noqa: BLE001 - a backend that raises while answering is not available
        return False


def available_backend() -> str | None:
    """The first accelerator this machine can actually use, or None.

    Asked backend by backend rather than through `torch.accelerator`, which reports what
    the build was compiled for rather than what works here - and the case this module is
    about is precisely a build that enumerates a card it cannot initialise.
    """
    for name in BACKENDS:
        if _backend_available(name):
            return name
    return None


def pick_device(requested: str | None = None):
    """The device a run will use: what was asked for, or the best available.

    Until `0.8.0a38` this asked about CUDA and nothing else, so an Apple Silicon Mac - every
    Mac sold since 2020 - trained on its processor with a working `mps` backend present in
    the same wheel. `requested` still wins over everything, because a run that says `cpu`
    means it.
    """
    import torch

    if requested:
        return torch.device(requested)
    backend = available_backend()
    return torch.device(backend if backend else "cpu")


def _nvidia_smi_sees_a_gpu() -> bool:
    """Whether the driver reports a card, which torch may still be unable to use.

    The last resort of the detection below, and the only one that works when torch cannot
    initialise CUDA at all.
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def device_name(backend: str) -> str:
    """What to call the thing the work will run on."""
    import torch

    if backend == "cuda":
        return torch.cuda.get_device_name(0)
    # The others have no name to ask for: there is one of each, built into the machine.
    return backend


def device_report() -> dict[str, Any]:
    """What this installation can see, as plain data.

    Returns:
        `torch` and `cuda_build` as strings; `backend`, the accelerator that will be used,
        or `None` for the processor; `device`, what to call it; `arch_list`, the
        architectures a CUDA build was compiled for; and `gpu_present_but_unusable`, which
        is the one worth acting on.

    `gpu_present_but_unusable` reads `device_count()` rather than `is_available()`, and
    that is the whole subtlety: with a CUDA 13 build on a Pascal card the first returns 1
    and the second False - torch enumerates the card and cannot initialise it, which is
    exactly the case to report. Testing for zero detects nothing. The driver is asked as
    well, because a build with no CUDA support at all enumerates nothing either way.
    """
    import torch

    backend = available_backend()
    # On its own line rather than inside the dict below: between the key's closing quote
    # and the next literal's opening one, `tests/test_public_api.py`'s scanner reads the
    # code as a quoted sentence and reports `device_name()` as a call this package tells
    # somebody to make. That test's own docstring records it catching its author once
    # already; this is the second shape, and the cheaper fix is here.
    name = device_name(backend) if backend else "cpu"
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda or "cpu-only build",
        "cuda_available": backend == "cuda",
        "backend": backend,
        "device": name,
        "arch_list": list(torch.cuda.get_arch_list()) if backend == "cuda" else [],
    }
    if backend is None:
        try:
            count = torch.cuda.device_count()
        except Exception:  # noqa: BLE001
            count = 0
        report["gpu_present_but_unusable"] = count > 0 or _nvidia_smi_sees_a_gpu()
    return report
