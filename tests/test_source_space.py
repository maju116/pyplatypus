"""Giving the answer back on the grid the data arrived on.

The model works at its own size; a mask is only useful on the scan it describes. These tests
are about the round trip, and the one that matters is
`test_the_returned_mask_covers_the_anatomy_in_the_source_scan`: the shape being right proves
nothing, because a mask can have exactly the right shape and describe the wrong place.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyplatypus import Engine, from_dict
from pyplatypus.engine import EngineError

nib = pytest.importorskip("nibabel")


def write_case(root, name, shape, spacing, block):
    """One case: a bright block in air, and a label map marking exactly that block."""
    sample = root / name
    (sample / "images").mkdir(parents=True, exist_ok=True)
    (sample / "masks").mkdir(parents=True, exist_ok=True)

    labels = np.zeros(shape, dtype=np.float32)
    labels[block] = 1
    scan = np.where(labels > 0, 40.0, -1000.0).astype(np.float32)
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(scan, affine), str(sample / "images" / "ct.nii.gz"))
    nib.save(nib.Nifti1Image(labels, affine), str(sample / "masks" / "seg.nii.gz"))
    return sample


def engine_for(root, **data):
    block = {"train_path": str(root), "validation_path": str(root), "labels": [0, 1],
             "window": "soft_tissue", "shuffle": False}
    block.update(data)
    spec = from_dict({
        "data": block,
        "models": [{"name": "m", "input_shape": [32, 32, 32], "n_class": 2, "channels": 1,
                    "blocks": 2, "filters": 4, "batch_size": 1, "epochs": 1,
                    "metrics": [{"name": "dice"}]}],
    })
    engine = Engine(spec, device="cpu")
    engine.fit()
    return engine


# --------------------------------------------------------------------- shapes
def test_model_space_is_one_stacked_array(tmp_path):
    root = tmp_path / "cases"
    write_case(root, "a", (48, 48, 24), (1.0, 1.0, 2.0), np.s_[12:36, 12:36, 6:18])
    write_case(root, "b", (48, 48, 24), (1.0, 1.0, 2.0), np.s_[12:36, 12:36, 6:18])

    engine = engine_for(root, target_spacing=(1.0, 1.0, 1.0))
    predictions = engine.predict("m", split="validation")

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2, 32, 32, 32, 2)


def test_source_space_is_a_list_on_each_scan_s_own_grid(tmp_path):
    """Different sources have different shapes, so this cannot be one array - and saying so
    through the return type beats resizing them to match, which would put a mask on anatomy it
    was not computed from."""
    root = tmp_path / "cases"
    write_case(root, "small", (48, 48, 24), (1.0, 1.0, 2.0), np.s_[12:36, 12:36, 6:18])
    write_case(root, "large", (64, 64, 40), (1.0, 1.0, 1.5), np.s_[16:48, 16:48, 10:30])

    engine = engine_for(root, target_spacing=(1.0, 1.0, 1.0))
    predictions = engine.predict("m", split="validation", space="source")

    assert isinstance(predictions, list)
    # Paired by sample rather than by position: samples are discovered in sorted order, so
    # "large" comes before "small", and asserting a guessed order tests the alphabet.
    shapes = {
        sample.key: prediction.shape
        for sample, prediction in zip(engine._samples["validation"], predictions, strict=True)
    }
    assert shapes == {"small": (48, 48, 24, 2), "large": (64, 64, 40, 2)}


def test_source_space_works_without_resampling(tmp_path):
    """The gap is not only about `target_spacing`: without it the volume is resized into the
    model's shape, and the answer still belongs back at the scan's size."""
    root = tmp_path / "cases"
    write_case(root, "a", (48, 48, 24), (1.0, 1.0, 2.0), np.s_[12:36, 12:36, 6:18])

    engine = engine_for(root)
    predictions = engine.predict("m", split="validation", space="source")
    assert predictions[0].shape == (48, 48, 24, 2)


def test_a_source_already_at_the_model_s_shape_is_returned_untouched(tmp_path):
    root = tmp_path / "cases"
    write_case(root, "a", (32, 32, 32), (1.0, 1.0, 1.0), np.s_[8:24, 8:24, 8:24])

    engine = engine_for(root, target_spacing=(1.0, 1.0, 1.0))
    model = engine.predict("m", split="validation")
    source = engine.predict("m", split="validation", space="source")
    assert np.array_equal(source[0], model[0])


# ----------------------------------------------------------- the real question
def test_the_returned_mask_covers_the_anatomy_in_the_source_scan(tmp_path):
    """Shape is not the claim; position is.

    The scan holds a bright block in a known place. A mask that came back on the scan's grid has
    to mark *that* block - checked against the scan's own voxels, not against another
    prediction. A version of this that only compared shapes would pass with the mask upside
    down.
    """
    root = tmp_path / "cases"
    block = np.s_[8:20, 24:44, 4:14]          # deliberately off-centre and not a cube
    for name in ("a", "b", "c", "d"):
        write_case(root, name, (48, 48, 24), (1.0, 1.0, 2.0), block)

    spec = from_dict({
        "data": {"train_path": str(root), "validation_path": str(root), "labels": [0, 1],
                 "window": "soft_tissue", "target_spacing": (1.0, 1.0, 1.0),
                 "shuffle": False},
        "models": [{"name": "m", "input_shape": [48, 48, 48], "n_class": 2, "channels": 1,
                    "blocks": 2, "filters": 8, "batch_size": 2, "epochs": 20,
                    "loss": {"name": "dice"},
                    "metrics": [{"name": "dice", "include_background": False}]}],
    })
    engine = Engine(spec, device="cpu")
    engine.fit()

    predictions = engine.predict("m", split="validation", space="source")
    classes = predictions[0].argmax(axis=-1)

    truth = np.zeros((48, 48, 24), dtype=bool)
    truth[block] = True

    # Overlap with the truth in the *source* grid. A mask that is the right size but in the
    # wrong place scores near zero here and passes every shape assertion there is.
    #
    # Threshold well below what this reaches - measured 0.97 at twenty epochs, 1.00 at thirty -
    # because the point is position, not convergence, and a random initialisation should not
    # make the test flaky. Twelve epochs reached 0.53, so the run does have to be long enough to
    # learn something: a model that learned nothing would also fail this, for the wrong reason.
    intersection = int((classes.astype(bool) & truth).sum())
    dice = 2 * intersection / (int(classes.astype(bool).sum()) + int(truth.sum()))
    assert dice > 0.7


def test_the_round_trip_keeps_a_mask_where_it_was(tmp_path):
    """The transform on its own, without a model in the way.

    A prediction built to mark exactly the block, taken through the inverse, must still mark it
    in the source grid. This isolates the geometry from whatever the network learned - which
    matters, because a badly trained model would hide a broken inverse and vice versa.
    """
    root = tmp_path / "cases"
    block = np.s_[8:20, 24:44, 4:14]
    write_case(root, "a", (48, 48, 24), (1.0, 1.0, 2.0), block)

    # The model's grid covers the whole resampled volume - 48 x 48 x 24 at 2 mm slices becomes
    # 48 x 48 x 48 at 1 mm - so nothing is cropped and the geometry is the only thing under
    # test. With a smaller input_shape most of this block falls outside the crop and the round
    # trip legitimately loses it, which is what the first version of this test measured: 0.89,
    # and honest.
    spec = from_dict({
        "data": {"train_path": str(root), "validation_path": str(root), "labels": [0, 1],
                 "window": "soft_tissue", "target_spacing": (1.0, 1.0, 1.0),
                 "shuffle": False},
        "models": [{"name": "m", "input_shape": [48, 48, 48], "n_class": 2, "channels": 1,
                    "blocks": 2, "filters": 4, "batch_size": 1, "epochs": 1}],
    })
    engine = Engine(spec, device="cpu")
    engine.fit()
    sample = engine._samples["validation"][0]
    model_spec = engine.spec.models[0]

    # What the dataset hands the network for this sample, as a perfect prediction: the mask it
    # was given, one-hot, on the model's grid.
    dataset = engine.dataset(model_spec, "validation")
    _, mask = dataset[0]
    back = engine._to_source_space(mask.astype(np.float32), sample, model_spec)

    classes = back.argmax(axis=-1).astype(bool)
    truth = np.zeros((48, 48, 24), dtype=bool)
    truth[block] = True

    intersection = int((classes & truth).sum())
    dice = 2 * intersection / (int(classes.sum()) + int(truth.sum()))
    # Interpolation softens the boundary, so this is not exact - but it is close, and a broken
    # inverse would be nowhere near.
    assert dice > 0.9


def test_a_cropped_volume_comes_back_padded_with_background(tmp_path):
    """Where the forward crop cut anatomy away, the inverse pads with background.

    That padding means *not examined* rather than *nothing there*, which is worth knowing and is
    documented on the method. What must not happen is a shape mismatch or an error: the caller
    asked for a mask for their scan and gets one, of the right size, with the parts the model
    never saw marked as background.
    """
    root = tmp_path / "cases"
    # 96 voxels across, model works at 32 after resampling: most of this is cropped away.
    write_case(root, "wide", (96, 96, 32), (1.0, 1.0, 1.0), np.s_[40:56, 40:56, 12:20])

    engine = engine_for(root, target_spacing=(1.0, 1.0, 1.0))
    prediction = engine.predict("m", split="validation", space="source")[0]

    assert prediction.shape == (96, 96, 32, 2)
    # The edges were never examined, so they are background - not an error and not a lesion.
    assert prediction[0, 0, 0].argmax() == 0


# --------------------------------------------------------------------- refusals
def test_an_unknown_space_is_refused(tmp_path):
    root = tmp_path / "cases"
    write_case(root, "a", (48, 48, 24), (1.0, 1.0, 2.0), np.s_[12:36, 12:36, 6:18])
    engine = engine_for(root)
    with pytest.raises(EngineError, match="'model' or 'source'"):
        engine.predict("m", split="validation", space="patient")
