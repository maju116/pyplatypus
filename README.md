<img src="https://raw.githubusercontent.com/maju116/platypus/master/man/figures/hexsticker_platypus.png" align="right" alt="" width="130" />

# pyplatypus

**Computer vision for medical imaging — the engine behind the `platypus` R package.**

> **0.3.0a10 — an alpha.** This replaces the 2022 TensorFlow package with a PyTorch one.
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
- **Resampling to a common voxel size**: `target_spacing: [1, 1, 1]` fixes the millimetres
  per voxel and then centre-crops or pads to the model's `input_shape`, instead of squeezing
  every scan into the same box. Clinical scans cover whatever length the question needed - 40
  slices of 1 mm is 40 mm of patient, 40 slices of 2.5 mm is 100 mm - so resizing alone makes
  the same organ a different size in each and the network cannot tell.
- **One channel per file, in a stated order**: `channels_from` takes one pattern per channel,
  which is how BraTS ships four MRI sequences per patient and how satellite sets keep their
  bands. Sorting the names gives flair, t1, t1ce, t2 - reproducible, and anatomically
  meaningless; a model trained with FLAIR in channel one and used where channel one is T1
  answers plausibly and reports nothing. A pattern matching two files, or none, or leaving a
  file unused, is an error.
- **The answer on the grid it came from**: `predict(space="source")` maps each prediction back
  onto the scan it was computed from, undoing the resampling and the crop. A mask on the model's
  grid cannot be laid over the patient's scan by anything, which is the whole point of producing
  one. Returns a list rather than an array, because scans differ in size and resizing them to
  match is how a mask ends up describing the wrong anatomy.
- **Weights by name, pinned to a commit**: `weights: dsbowl-unet` is published and fetches
  from a registry entry that carries the exact commit they were published at, so a name means one
  set of numbers forever. `hf://owner/repo/file.safetensors@commit` takes anything else on the
  Hub, a path takes a local file, and a sidecar records what the weights were trained for -
  loading refuses a model they do not belong to, because weights with the wrong colormap and the
  right shape load cleanly and predict nonsense.
- **Many models from one file**, with a comparison table at the end - which is how you find
  out that on some problems the architecture is not what matters. See below.
- **A pretrained backbone as the contracting path**: `encoder: resnet34` builds that
  architecture, and `pretrained: true` is the separate flag that loads its ImageNet weights -
  separate because the first thing a named backbone should not do is reach the network on a
  machine that has none. Any timm backbone whose features start at half resolution works;
  `blocks` says how many of its stages to use, and the full-resolution level a U-shaped
  decoder needs is one block of ours, because no ImageNet stem has one. What it is measurably
  worth, and what it is not, is below.

Augmentation works in 3D, with a caveat the package handles rather than hides: albumentations
supports volumes unevenly - 97 of its transforms take one and the rest raise from inside the
library, `GaussNoise` as `KeyError: 'images'`. Every transform in a 3D specification is tried
against a small probe volume while the pipeline is built, so an unsupported one is named before
training starts, and `available_transforms(rank=3)` lists what is usable.

Object detection and ensembling remain out of scope.

Resampling is opt-in rather than automatic: it changes the voxel grid the model sees, which
is a decision to take deliberately. Without `target_spacing` the old behaviour stands and
volumes are resized to `input_shape`.

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

## Many models, one table

The point of a specification holding several models is that the comparison costs one command. Run
`examples/compare_dsbowl_architectures.py` and all four architectures train on the same split with
the same augmentation:

| model | parameters | s/epoch | epochs | Dice mean | sd | median | min |
|---|---|---|---|---|---|---|---|
| U-Net | 1,942,594 | 15.5 | 60 | 0.9217 | 0.054 | 0.9353 | 0.7227 |
| U-Net++ | 2,263,730 | 18.8 | 44 | 0.9212 | 0.055 | 0.9323 | 0.7459 |
| LinkNet | 1,746,754 | 14.0 | 60 | 0.9203 | 0.053 | 0.9293 | 0.7205 |
| Res-U-Net | 2,029,682 | 14.2 | 55 | 0.9187 | 0.058 | 0.9293 | 0.6706 |

Which is worth reading for what it says rather than for the winner. **The spread across the four
is 0.0030, and the spread across images within any one of them is 0.054** — the architectures are
eighteen times closer to each other than the images are. Re-running the same U-Net with a
different seed moves the number by 0.0012, which is 40% of the whole spread.

On this problem the architecture is not where the result comes from, and a table like this is how
you find that out in an hour instead of a fortnight. That is also why only one set of weights is
published: four names that mean the same thing would be four promises nobody needed.

## What a pretrained encoder is worth

`encoder` and `pretrained` are two flags, not one, and the reason is that the second is the only
one that reaches the network:

```yaml
models:
  - name: polyps
    encoder: resnet34        # builds a ResNet-34 contracting path
    pretrained: true         # and loads its ImageNet weights
    freeze_encoder: 5        # holding them still while the decoder settles
```

Measured on three datasets, one per modality, 3 seeds each, each configuration compared at its
best validation Dice with the best weights restored. Small training sets on purpose - 40 to 100
images - because that is the regime transfer learning is supposed to be for.

| | DS Bowl, microscopy | SIIM-ACR, chest X-ray | Kvasir-SEG, endoscopy |
|---|---|---|---|
| built-in encoder, 1.9M | **0.8460** ±0.0068 | 0.0531 ±0.0586 | 0.5830 ±0.0128 |
| resnet34 from scratch, 9.0M | 0.7891 ±0.0992 | 0.1073 ±0.0033 | 0.5670 ±0.0251 |
| pretrained, `freeze_encoder: 5` | 0.8333 ±0.0151 | **0.1203** ±0.0116 | **0.5932** ±0.0170 |
| pretrained, `encoder_learning_rate: lr/10` | 0.8326 ±0.0094 | 0.1011 ±0.0092 | 0.5329 ±0.0101 |

Four things come out of it, and they repeat on all three:

**Use `freeze_encoder`, not `encoder_learning_rate` alone.** Freezing is the better pretrained
variant everywhere, and on Kvasir a tenth of the learning rate is the *worst* configuration in the
table - below the built-in encoder and below the same ResNet trained from nothing. Giving one
learning rate to the whole network is worse still: on DS Bowl it costs 0.045 of Dice and
quadruples the seed spread, because a random decoder's gradients undo the transferred weights
before the decoder has learned anything to be worth it.

**Do not expect a better score. Expect to reach it sooner.** On Kvasir the built-in encoder needed
150, 110 and 83 epochs for its three seeds; pretrained-and-frozen needed 64, 82 and 83 for the
same Dice. That halving is the one benefit that showed up consistently.

**Capacity matters where the target is thin.** On SIIM, where a pneumothorax covers a median 0.59%
of the frame, every ResNet variant roughly doubles the built-in encoder's score - and the built-in
encoder's seed spread (±0.0586) is larger than its own mean, with two of three seeds never
learning at all. That is an argument for being able to swap the encoder, independent of ImageNet.

**ImageNet features do not transfer to these images.** The clearest number is from DS Bowl with
the backbone frozen for the whole run: 0.6068, against 0.8460 for an encoder that learned the
task from 40 images. And Kvasir is the fairest test available - camera photographs of tissue,
three channels, natural lighting - where the best pretrained configuration beats the built-in
encoder by 0.0102 with a seed spread of 0.0170. Inside the noise.

So the feature is here, correct, and documented for what it does: a faster route to the same
answer, and a bigger encoder for problems that need one. Our own 1.9M-parameter encoder beats
ResNet-34's 9.0M on two of the three datasets.

Reproduce with `examples/compare_pretrained_encoders.py`. Raw logs and per-seed numbers are in
`measurements/`. The SIIM-ACR images came from a Kaggle competition, so no weights trained on
them are published; BBBC038v1 is CC0 and Kvasir-SEG is CC BY 4.0.

## Requirements

Python ≥ 3.10, and torch ≥ 2.7.

**To fetch published weights by name**, install the `hub` extra:

```bash
pip install "pyplatypus[hub]"
```

It is not in the base install because most runs never fetch weights and an air-gapped one cannot.
Local weights files and everything else work without it.

**For a pretrained encoder**, install the `encoders` extra, which brings timm:

```bash
pip install "pyplatypus[encoders]"
```

On a Pascal-generation GPU ask for both extras together — `pyplatypus[encoders,pascal]`. timm
pulls in torchvision, torchvision pins its torch version by equality, and asking for timm alone
resolved torch to a CUDA 13 build whose kernels start at `sm_75`; a GTX 1070 is `sm_61`, so the
GPU disappears and the error blames the driver. The two extras together resolve to torch 2.7.1
and torchvision 0.22.1.

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
