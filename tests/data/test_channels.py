"""One channel per file, and the order nobody can afford to guess.

The test that matters is `test_channels_arrive_in_the_stated_order`: it puts a different value
in each modality and checks which channel each one landed in. Everything else here is about
refusing the situations where the order would be decided by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyplatypus.data.channels import ChannelError, match_channels

# BraTS names, which are the reason this exists: sorted, they come out flair, t1, t1ce, t2.
BRATS = [
    "case_001_t1.nii.gz",
    "case_001_t1ce.nii.gz",
    "case_001_t2.nii.gz",
    "case_001_flair.nii.gz",
]
PATTERNS = [r"_t1\.nii", r"_t1ce\.nii", r"_t2\.nii", r"_flair\.nii"]


def test_patterns_pick_the_files_in_channel_order(tmp_path):
    paths = [tmp_path / name for name in BRATS]
    ordered = match_channels(paths, PATTERNS)
    assert [p.name for p in ordered] == [
        "case_001_t1.nii.gz",
        "case_001_t1ce.nii.gz",
        "case_001_t2.nii.gz",
        "case_001_flair.nii.gz",
    ]


def test_the_sorted_order_is_not_the_wanted_order(tmp_path):
    """Stated rather than inferred, and this is why: the two disagree on real data."""
    assert sorted(BRATS) != BRATS
    assert [p.name for p in match_channels([tmp_path / n for n in BRATS], PATTERNS)] != sorted(
        BRATS
    )


def test_a_pattern_matching_two_files_is_refused(tmp_path):
    """`_t1` matches `_t1.nii.gz` and `_t1ce.nii.gz`. Taking the first would make the channel
    order depend on directory listing order, which is the bug this module exists to prevent."""
    loose = [r"_t1", r"_t1ce\.nii", r"_t2\.nii", r"_flair\.nii"]
    with pytest.raises(ChannelError, match="matches 2 files"):
        match_channels([tmp_path / n for n in BRATS], loose)


def test_a_pattern_matching_nothing_names_the_files_present(tmp_path):
    with pytest.raises(ChannelError, match="matches no file"):
        match_channels([tmp_path / n for n in BRATS], [r"_t1\.nii", r"_dwi\.nii"])


def test_the_sample_is_named_in_the_complaint(tmp_path):
    with pytest.raises(ChannelError, match="sample 'case_001'"):
        match_channels([tmp_path / n for n in BRATS], [r"_t1\.nii", r"_dwi\.nii"],
                       key="case_001")


def test_a_file_matching_no_channel_is_refused(tmp_path):
    """Left out silently it would be data the model never sees - and on a dataset where an
    extra sequence appears for some patients only, that is a difference between cases."""
    files = [*BRATS, "case_001_dwi.nii.gz"]
    with pytest.raises(ChannelError, match="match no channel pattern"):
        match_channels([tmp_path / n for n in files], PATTERNS)


def test_an_invalid_regular_expression_says_so(tmp_path):
    with pytest.raises(ChannelError, match="not a valid regular expression"):
        match_channels([tmp_path / n for n in BRATS], [r"_t1\.nii", r"_t2(nii"])


# ------------------------------------------------------------ through the spec
def brats_case(root, name, values, spacing=(1.0, 1.0, 1.0), shape=(16, 16, 8)):
    """One patient: four sequences, each filled with its own value, and a label map."""
    nib = pytest.importorskip("nibabel")
    sample = root / name
    (sample / "images").mkdir(parents=True, exist_ok=True)
    (sample / "masks").mkdir(parents=True, exist_ok=True)
    affine = np.diag([*spacing, 1.0])
    for sequence, value in values.items():
        nib.save(nib.Nifti1Image(np.full(shape, float(value), dtype=np.float32), affine),
                 str(sample / "images" / f"{name}_{sequence}.nii.gz"))
    labels = np.zeros(shape, dtype=np.float32)
    labels[4:12, 4:12, 2:6] = 1
    nib.save(nib.Nifti1Image(labels, affine), str(sample / "masks" / f"{name}_seg.nii.gz"))
    return sample


def build_dataset(root, **data_kwargs):
    from pyplatypus.data.dataset import SegmentationDataset
    from pyplatypus.data.paths import discover_samples
    from pyplatypus.spec.data import SegmentationData
    from pyplatypus.spec.models import SegmentationModel

    data = SegmentationData(train_path=str(root), validation_path=str(root), labels=[0, 1],
                            **data_kwargs)
    model = SegmentationModel(name="m", input_shape=(16, 16, 8), channels=4, 
                              blocks=2)
    return SegmentationDataset(discover_samples(root).samples, model, data)


def test_channels_arrive_in_the_stated_order(tmp_path):
    """The claim, checked by making each sequence identifiable.

    T1 holds 100, T1ce 200, T2 300, FLAIR 400 - so the channel a value lands in says which
    file it came from. The window is explicit: `full` takes each file's own range, and a file
    of one constant value has no range at all, so every channel would come back as zero. The
    first version of this test did exactly that and compared four zeros.
    """
    root = tmp_path / "cases"
    brats_case(root, "case_001", {"t1": 100, "t1ce": 200, "t2": 300, "flair": 400})

    dataset = build_dataset(root, channels_from=PATTERNS, window=(250.0, 500.0))
    image, mask = dataset[0]

    assert image.shape == (16, 16, 8, 4)
    means = [float(image[..., channel].mean()) for channel in range(4)]
    # Ascending, because 100 < 200 < 300 < 400 and that is the order the patterns asked for.
    assert means == sorted(means)
    assert len(set(means)) == 4
    assert mask.shape == (16, 16, 8, 2)


def test_the_channels_stay_matched_to_the_mask(tmp_path):
    root = tmp_path / "cases"
    brats_case(root, "case_001", {"t1": 100, "t1ce": 200, "t2": 300, "flair": 400})
    dataset = build_dataset(root, channels_from=PATTERNS, window=(250.0, 500.0))
    _, mask = dataset[0]
    # The mask marks a block; every voxel belongs to exactly one class whatever the channels do.
    assert float(mask.sum(axis=-1).min()) == 1.0


def test_channels_survive_resampling(tmp_path):
    """Each channel is resampled on its own, so they have to come back the same size or the
    stack cannot be built at all."""
    root = tmp_path / "cases"
    brats_case(root, "case_001", {"t1": 100, "t1ce": 200, "t2": 300, "flair": 400},
               spacing=(1.0, 1.0, 2.5))

    dataset = build_dataset(root, channels_from=PATTERNS, window=(250.0, 500.0),
                            target_spacing=(1.0, 1.0, 1.0))
    image, _ = dataset[0]
    assert image.shape == (16, 16, 8, 4)


def test_channels_of_different_shapes_are_refused(tmp_path):
    """Two sequences of one patient that do not line up voxel for voxel are not channels of one
    image - they are two images, and stacking them would put the second one's anatomy in the
    wrong place."""
    nib = pytest.importorskip("nibabel")
    root = tmp_path / "cases"
    sample = brats_case(root, "case_001", {"t1": 100, "t1ce": 200, "t2": 300, "flair": 400})
    # Rewrite one sequence at a different size.
    nib.save(nib.Nifti1Image(np.full((8, 8, 4), 200.0, dtype=np.float32), np.eye(4)),
             str(sample / "images" / "case_001_t1ce.nii.gz"))

    from pyplatypus.data.dataset import DataError

    dataset = build_dataset(root, channels_from=PATTERNS, window=(250.0, 500.0))
    with pytest.raises(DataError, match="different shapes"):
        dataset[0]


def test_a_case_missing_a_sequence_is_refused_by_name(tmp_path):
    """A dataset where one patient lacks a sequence is common, and training on it silently
    would train that patient on a duplicated or absent channel."""
    root = tmp_path / "cases"
    brats_case(root, "case_001", {"t1": 100, "t1ce": 200, "t2": 300, "flair": 400})
    brats_case(root, "case_002", {"t1": 100, "t1ce": 200, "t2": 300})

    dataset = build_dataset(root, channels_from=PATTERNS, window=(250.0, 500.0))
    dataset[0]                                   # the complete one is fine
    with pytest.raises(ChannelError, match="case_002"):
        dataset[1]


def test_a_multi_modal_spec_trains(tmp_path):
    from pyplatypus import Engine, from_dict

    root = tmp_path / "cases"
    for index in range(2):
        brats_case(root, f"case_{index:03d}",
                   {"t1": 100 + index, "t1ce": 200, "t2": 300, "flair": 400})

    engine = Engine(from_dict({
        "task": "semantic_segmentation",
        "data": {"train_path": str(root), "validation_path": str(root), "labels": [0, 1],
                 "channels_from": PATTERNS, "window": [250.0, 500.0],
                 "shuffle": False},
        "models": [{"name": "unet3d", "input_shape": [16, 16, 8], 
                    "channels": 4, "blocks": 2, "filters": 4, "batch_size": 1,
                    "epochs": 1, "metrics": [{"name": "dice"}]}],
    }), device="cpu")
    history = engine.fit()["unet3d"]

    assert len(history) == 1
    assert engine.predict("unet3d", split="validation").shape == (2, 16, 16, 8, 2)


def test_two_dimensional_channels_work_the_same_way(tmp_path):
    """Not only volumes: the old package's satellite example kept its bands in separate files
    too, and the rule is the same at rank 2."""
    from PIL import Image

    from pyplatypus.data.dataset import SegmentationDataset
    from pyplatypus.data.paths import discover_samples
    from pyplatypus.spec.data import SegmentationData
    from pyplatypus.spec.models import SegmentationModel

    root = tmp_path / "scene"
    sample = root / "tile_01"
    (sample / "images").mkdir(parents=True, exist_ok=True)
    (sample / "masks").mkdir(parents=True, exist_ok=True)
    for band, value in (("red", 40), ("green", 120), ("nir", 200)):
        Image.fromarray(np.full((16, 16), value, np.uint8)).save(
            sample / "images" / f"tile_01_{band}.png"
        )
    mask = np.zeros((16, 16, 3), np.uint8)
    mask[4:12, 4:12] = 255
    Image.fromarray(mask).save(sample / "masks" / "tile_01_mask.png")

    data = SegmentationData(train_path=str(root), validation_path=str(root),
                            colormap=[(0, 0, 0), (255, 255, 255)],
                            channels_from=[r"_red\.", r"_green\.", r"_nir\."])
    model = SegmentationModel(name="m", input_shape=(16, 16), channels=3, blocks=2)
    dataset = SegmentationDataset(discover_samples(root).samples, model, data)

    image, _ = dataset[0]
    assert image.shape == (16, 16, 3)
    means = [float(image[..., channel].mean()) for channel in range(3)]
    assert means == sorted(means)                # red < green < nir, as asked for
