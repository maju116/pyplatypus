"""Where trained weights come from, and what has to be true before they are loaded.

Three ways to name weights, in the order they will be used:

    weights: dsbowl-unet                                  a name from the registry below
    weights: hf://owner/repo/file.safetensors@a1b2c3d      anything else on the Hub
    weights: /home/me/run-07.safetensors                   a local file

**Every reference resolves to one immutable set of numbers.** A registry name carries the
commit it was published at, and the `hf://` form requires one, because weights that change
under a stable name are the worst kind of irreproducibility: the code is identical, the
result is not, and nothing in either says why. This is the same reasoning as the R package
pinning an exact engine.

**A sidecar describes what the weights are for.** Beside every published file sits a JSON
recording the architecture, input shape, channel count and class count they were trained
with, and loading refuses when the model asks for something else. Without it, a state dict
with the wrong number of classes is a shape error deep in torch at best, and at worst - when
the shapes happen to line up - a model that predicts confidently and means nothing.

**safetensors, not pickle.** A pickle executes code on load, so a weights file is a program
someone else wrote. The Hub flags them for exactly this reason. safetensors is data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pyplatypus.errors import PlatypusError

SIDECAR_SUFFIX = ".json"
HF_PATTERN = re.compile(
    r"^hf://(?P<owner>[^/]+)/(?P<repo>[^/]+)/(?P<path>.+?)(?:@(?P<revision>[^@/]+))?$"
)


class WeightsError(PlatypusError):
    kind = "weights_error"


@dataclass(frozen=True)
class Published:
    """One entry in the registry: a file, in a repo, at a commit."""

    repo: str
    filename: str
    revision: str
    description: str

    @property
    def reference(self) -> str:
        return f"hf://{self.repo}/{self.filename}@{self.revision}"


# The registry. Deliberately small and curated: a name here is a promise that the numbers
# behind it will not change, so adding one is a decision and not a side effect of training
# something. Revisions are full commit hashes rather than tags - a tag can be moved.
REGISTRY: dict[str, Published] = {
    "dsbowl-unet": Published(
        repo="maju116/platypus-weights",
        filename="dsbowl-unet.safetensors",
        revision="5e035c6a39827c33e19507affb9415c3908ae383",
        description=(
            "U-Net, 256x256, nuclei in light microscopy. Trained on BBBC038v1 (the 2018 Data "
            "Science Bowl, CC0). Dice 0.92 mean over 134 held-out images, worst case 0.72. "
            "Semantic, not instance: touching nuclei come back as one region."
        ),
    ),
}


def known_weights() -> dict[str, str]:
    """Registry names and what they are, for listing to a user."""
    return {name: entry.description for name, entry in REGISTRY.items()}


def resolve_weights(reference: str) -> Path:
    """Turn any of the three forms into a local file, downloading if it has to.

    Downloads land in the Hugging Face cache - `~/.cache/huggingface/hub` unless `HF_HOME`
    says otherwise - which is outside any virtual environment, so the file survives an
    environment being rebuilt. Asking twice costs nothing: the cache is addressed by content
    and a pinned commit needs no network call once it is there.
    """
    if reference.startswith("hf://"):
        return _from_hub(reference)
    if reference in REGISTRY:
        return _from_hub(REGISTRY[reference].reference)

    path = Path(reference).expanduser()
    if path.exists():
        return path

    # Not a path that exists, not a name we know. Which of those it was meant to be decides
    # what the user needs to hear.
    if "/" in reference or reference.endswith((".safetensors", ".pt", ".pth")):
        raise WeightsError(
            f"'{reference}' is not a file. For weights from the Hub write "
            "hf://owner/repo/file.safetensors@commit, and note that the commit is required."
        )
    listed = ", ".join(sorted(REGISTRY)) or "none published yet"
    raise WeightsError(
        f"no published weights called '{reference}'. Known names: {listed}. You can also "
        "give a path to a local file, or hf://owner/repo/file.safetensors@commit."
    )


def _from_hub(reference: str) -> Path:
    found = HF_PATTERN.match(reference)
    if found is None:
        raise WeightsError(
            f"'{reference}' is not a Hub reference. The form is "
            "hf://owner/repo/file.safetensors@commit."
        )
    owner, repo, path, revision = found.group("owner", "repo", "path", "revision")
    if not revision:
        raise WeightsError(
            f"'{reference}' names no commit. Add @commit: without it the same reference can "
            "mean different numbers next month, which makes a result impossible to reproduce "
            "and impossible to debug. `main` is not a version."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise WeightsError(
            "downloading weights needs the huggingface_hub package, which this package does "
            "not install by default - most runs never fetch weights and an air-gapped one "
            "cannot. Install it with `pip install pyplatypus[hub]`, or download the file "
            "yourself and give the path instead."
        ) from None

    repo_id = f"{owner}/{repo}"
    try:
        local = hf_hub_download(repo_id=repo_id, filename=path, revision=revision)
    except Exception as error:  # noqa: BLE001 - huggingface_hub raises a family of its own
        raise WeightsError(
            f"could not fetch {path} from {repo_id} at {revision}: "
            f"{type(error).__name__}: {error}"
        ) from None

    sidecar = _fetch_sidecar(repo_id, path, revision)
    if sidecar is not None:
        _remember_sidecar(Path(local), sidecar)
    return Path(local)


def _fetch_sidecar(repo_id: str, path: str, revision: str) -> dict | None:
    """The description beside the weights, if the repo carries one.

    Optional on purpose: someone else's weights need not follow our convention, and refusing
    them for that would make the `hf://` form useless. Ours always carry it.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    stem = path.rsplit(".", 1)[0]
    try:
        local = hf_hub_download(repo_id=repo_id, filename=f"{stem}{SIDECAR_SUFFIX}",
                                revision=revision)
    except EntryNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - a missing description must not fail a download
        return None
    return json.loads(Path(local).read_text())


_SIDECARS: dict[str, dict] = {}


def _remember_sidecar(weights: Path, sidecar: dict) -> None:
    _SIDECARS[str(weights)] = sidecar


def describe(weights: Path) -> dict | None:
    """What a weights file says about itself, from its sidecar.

    Looked for beside the file first - that is where `export_weights` puts it - and then in
    what a download recorded, since the Hub cache stores files under their hashes and the
    sidecar does not sit next to the weights there.
    """
    beside = weights.with_suffix(SIDECAR_SUFFIX)
    if beside.exists():
        return json.loads(beside.read_text())
    return _SIDECARS.get(str(weights))


def load_into(model, reference: str, spec) -> dict | None:
    """Load weights into a model, after checking they belong to it.

    Returns the sidecar when there was one, so a caller can report what it loaded.
    """
    import torch

    path = resolve_weights(reference)
    sidecar = describe(path)
    if sidecar is not None:
        _refuse_mismatch(sidecar, spec, reference)

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError:  # pragma: no cover - safetensors is a hard dependency
            raise WeightsError(
                "reading .safetensors needs the safetensors package, which should have been "
                "installed with this one."
            ) from None
        state = load_file(str(path))
    else:
        # A checkpoint this package wrote with torch.save. weights_only=True so that loading
        # a file cannot execute code that came with it.
        state = torch.load(str(path), map_location="cpu", weights_only=True)

    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise WeightsError(
            f"'{reference}' does not fit this model: {error}. Weights belong to the exact "
            "architecture they were trained with - same blocks, same filters, same rank."
        ) from None
    return sidecar


def _refuse_mismatch(sidecar: dict, spec, reference: str) -> None:
    """Compare what the weights are for with what the model is, and say which field differs.

    Before loading rather than after: `load_state_dict` catches a different number of
    channels because the shapes disagree, but it cannot catch weights trained on a different
    colormap with the same class count. Those load cleanly and predict nonsense.
    """
    problems = []
    for field, mine in spec.weights_fingerprint().items():
        theirs = sidecar.get(field)
        if theirs is None:
            continue
        if isinstance(mine, list):
            # Compared as lists, because that is what JSON gives back, but rendered as
            # tuples: a shape reads as (256, 256) in Python and in every message this
            # package has ever printed, and a released vignette shows it that way.
            theirs = list(theirs)
            if theirs != mine:
                problems.append(f"{field}: weights say {tuple(theirs)}, the model says "
                                f"{tuple(mine)}")
            continue
        if theirs != mine:
            problems.append(f"{field}: weights say {theirs}, the model says {mine}")

    if problems:
        listed = "\n  - ".join(problems)
        raise WeightsError(
            f"'{reference}' was trained for a different model:\n  - {listed}\n"
            "Loading anyway would either fail inside torch or, where the shapes happen to "
            "agree, produce a model that predicts confidently and means nothing."
        )


def export_weights(model, spec, path: str | Path, *, extra: dict | None = None) -> Path:
    """Write a model's weights as safetensors, with the sidecar that describes them.

    The sidecar is not optional here. Weights without a record of what they are for are the
    thing this module exists to prevent, and the moment to write it is while the information
    is still at hand.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:  # pragma: no cover
        raise WeightsError(
            "writing .safetensors needs the safetensors package, which should have been "
            "installed with this one."
        ) from None

    target = Path(path).expanduser()
    if target.suffix != ".safetensors":
        target = target.with_suffix(".safetensors")
    target.parent.mkdir(parents=True, exist_ok=True)

    # contiguous() because safetensors refuses a view, and a state dict can hold them.
    state = {name: value.detach().cpu().contiguous()
             for name, value in model.state_dict().items()}
    save_file(state, str(target))

    sidecar = {
        **spec.weights_fingerprint(),
        # Two different counts, named apart. `parameters` is what the comparison table reports;
        # a state dict also holds buffers - BatchNorm's running statistics - so summing it gives
        # a larger number. Having both under one name made the sidecar disagree with
        # `evaluate()` by 2,962 for no stated reason.
        "parameters": sum(value.numel() for value in model.parameters()),
        "state_dict_numel": sum(value.numel() for value in state.values()),
    }
    if extra:
        sidecar.update(extra)
    target.with_suffix(SIDECAR_SUFFIX).write_text(json.dumps(sidecar, indent=2) + "\n")
    return target
