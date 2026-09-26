<img src="https://raw.githubusercontent.com/maju116/platypus/master/man/figures/hexsticker_platypus.png" align="right" alt="" width="130" />

# pyplatypus

**Computer vision for medical imaging — the engine behind the `platypus` R package.**

> **0.3.0a1 — an alpha.** This replaces the 2022 TensorFlow package with a PyTorch one.
> The API will still move and the R surface does not exist yet, so pin the exact version
> if you build on it.
>
> Everything on PyPI so far is a pre-release, so `pip install pyplatypus` resolves to this
> one. `pip install pyplatypus==0.1.0rc2` gets the old TensorFlow package.

## What works today

Semantic segmentation in 2D, end to end:

- **One spec, two ways in.** Build it from arguments or load it from YAML — both produce
  the same object, so nothing downstream can tell which you used.
- **Four architectures**: U-Net, U-Net++, Res-U-Net, LinkNet, each composable with
  separable convolutions, spatial dropout, learned or interpolated upsampling, deep
  supervision and configurable block width.
- **Nine losses** (IoU, Dice, CCE, CCE-Dice, Focal, Tversky, Focal-Tversky, Combo,
  Lovász) and three metrics, all reducing over every axis except batch and channel — so
  they already work on volumes.
- **Tiling that goes both ways**: cut a large image into a grid instead of shrinking it,
  and get a full-size mask back.
- **DICOM, read properly**: stored values converted to real units through the modality
  LUT, a fixed window rather than one taken from the image, and MONOCHROME1 inverted.
  Opening the file is the easy part; those three are what make two scans comparable.
- **Splitting by patient, not by slice**: `split_dataset()` turns one directory into
  train/validation/test CSVs, keeping every group whole. Slices of one patient on both
  sides of a split is the most common way to publish a Dice that means nothing, and a
  `group_by` that matches nothing is an error rather than a silent fallback.
- **Scores per case, not one number**: `evaluate_cases()` reports each image or patient
  separately and `summarise_cases()` gives the distribution and names the worst one. On the
  Data Science Bowl a model at 0.855 over the split turned out to score 0.006 on three
  images; the mean had no way of saying so.
- **Volumes, not only slices**: NIfTI in, 3D U-Net out, patches through the same tiling
  that cuts 2D images. Every volume is reoriented to canonical (RAS) first - a NIfTI's
  affine, not its array order, says where the anatomy is, and two datasets read naively can
  be mirror images of each other. Masks may be label maps (`labels: [0, 1]`) as well as
  pictures, because that is how every volume format stores them.
- **Many models from one file**, with a comparison table at the end.

Object detection, ensembling and pretrained backbones remain out of scope. So does
augmentation in 3D: albumentations ships 3D transforms through a different call signature,
and a 3D spec asking for augmentation is refused rather than silently ignored.

Also not done yet, deliberately: **resampling to isotropic spacing**. Spacing is read and
carried (`volume_spacing()`), because losing it would make resampling impossible later, but
applying it changes the voxel grid the model sees and that is a decision to make explicitly
rather than inside a reader.

## Try it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest

# The linter is pinned in pyproject.toml and installed on its own, so it agrees with CI
# and does not drag torch along to read text files.
uv run --only-group lint ruff check pyplatypus tests
```

```python
from pyplatypus import Engine, from_yaml

engine = Engine(from_yaml("examples/data_science_bowl.yaml"))
engine.fit(verbose=True)

for row in engine.evaluate():
    print(row)

masks = engine.predict(engine.best_model("dice"), split="test")
```

`examples/data_science_bowl.yaml` trains a U-Net and a LinkNet on the 2018 Data Science
Bowl and prints a comparison. On a GTX 1070 that is about 11 seconds per epoch at
160×160.

## Requirements

Python ≥ 3.10, and torch ≥ 2.7.

**If your GPU is a GTX 10-series (Pascal) or older**, install the `pascal` extra:

```bash
pip install "pyplatypus[pascal]"
```

torch 2.8 and later ship CUDA 13 builds, and CUDA 13 dropped the Maxwell, Pascal and
Volta generations outright - no driver update brings them back. The last torch built
against CUDA 12 is 2.7.x, which the extra pins. On anything from Turing (RTX 20-series)
onwards, ignore this.

## Licence

MIT.
