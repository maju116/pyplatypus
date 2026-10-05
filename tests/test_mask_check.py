"""Refusing to train on masks the colormap does not describe.

This is the failure the engine could not see until now: a colormap that matches none of
the labelled tissue trains without complaint, the loss falls because the background is
most of every medical image, and the model learns to answer "background" everywhere.

The check is built on class presence rather than on the unmatched fraction, and the tests
below are mostly about why: the unmatched fraction **scales with the size of the thing
being segmented**, so for a small lesion - the ordinary medical case - the most dangerous
mistake produces the quietest signal.
"""

import warnings

import numpy as np
import pytest
from PIL import Image

from pyplatypus import Engine, from_dict
from pyplatypus.engine import EngineError


def dataset_with_foreground(root, fraction, *, n=6, size=64, colour=(255, 255, 255)):
    """`n` samples whose foreground covers about `fraction` of each image."""
    side = max(1, round((fraction * size * size) ** 0.5))
    for index in range(n):
        sample = root / f"case_{index:02d}"
        (sample / "images").mkdir(parents=True, exist_ok=True)
        (sample / "masks").mkdir(parents=True, exist_ok=True)
        mask = np.zeros((size, size, 3), np.uint8)
        mask[5:5 + side, 5:5 + side] = colour
        Image.fromarray(mask).save(sample / "masks" / "m.png")
        image = np.clip(mask[..., :1].repeat(3, axis=2) * 0.6 + 20, 0, 255)
        Image.fromarray(image.astype(np.uint8)).save(sample / "images" / "a.png")
    return root


def spec_for(root, colormap, **overrides):
    return from_dict({
        "task": "semantic_segmentation",
        "data": {"train_path": str(root), "validation_path": str(root),
                 "colormap": colormap, "mode": "nested_dirs"},
        "models": [{"name": "m", "input_shape": [64, 64], "channels": 3,
                    "n_class": len(colormap), "blocks": 4, "filters": 8,
                    "epochs": 1, "batch_size": 2, **overrides}],
    })


# --- the reason the check is not a threshold on the unmatched fraction ------------------

@pytest.mark.parametrize("fraction", [0.25, 0.05, 0.004])
def test_the_unmatched_fraction_scales_with_the_foreground(tmp_path, fraction):
    """So it cannot be thresholded: at 0.4% a completely wrong colormap is quieter than
    the JPEG compression around a correct mask's edge, which measures about 0.7%."""
    root = dataset_with_foreground(tmp_path / f"f{fraction}", fraction)
    spec = spec_for(root, [[0, 0, 0], [128, 0, 0]])          # white masks, red asked for
    report = Engine(spec, check_masks=False).dataset(spec.models[0], "train").inspect_masks()

    assert report.unmatched == pytest.approx(fraction, rel=0.35)
    # Whatever the size, the class is simply not there - and that does not scale.
    assert report.missing_classes == [1]


def test_a_correct_colormap_matches_everything(tmp_path):
    root = dataset_with_foreground(tmp_path / "ok", 0.05)
    spec = spec_for(root, [[0, 0, 0], [255, 255, 255]])
    report = Engine(spec).dataset(spec.models[0], "train").inspect_masks()

    assert report.unmatched == 0
    assert report.missing_classes == []
    assert report.present_classes == [0, 1]


def test_a_class_the_data_does_not_have_is_invisible_to_the_fraction(tmp_path):
    """The case that makes the point: n_class=3 on binary masks. Every voxel matches an
    entry, so the unmatched fraction is zero and says everything is fine."""
    root = dataset_with_foreground(tmp_path / "three", 0.05)
    spec = spec_for(root, [[0, 0, 0], [255, 255, 255], [128, 0, 0]])
    report = Engine(spec, check_masks=False).dataset(spec.models[0], "train").inspect_masks()

    assert report.unmatched == 0
    assert report.missing_classes == [2]


# --- what fit() does with it -----------------------------------------------------------

def test_fit_refuses_a_colormap_that_matches_no_tissue(tmp_path):
    root = dataset_with_foreground(tmp_path / "wrong", 0.004)
    with pytest.raises(EngineError) as raised:
        Engine(spec_for(root, [[0, 0, 0], [128, 0, 0]])).fit()

    message = str(raised.value)
    assert "class 1 (colour (128, 0, 0))" in message
    assert "never appears" in message
    assert "checked 6 of 6" in message
    # And it says what to do, rather than only what is wrong.
    assert "check_masks=False" in message
    assert "inspect_masks" in message


def test_the_refusal_names_a_label_when_labels_were_given(volume_root):
    """A label map is the other way in, and a class is named by its label there."""
    spec = from_dict({
        "task": "semantic_segmentation",
        "data": {"train_path": str(volume_root), "validation_path": str(volume_root),
                 "labels": [0, 1, 7], "mode": "nested_dirs"},
        "models": [{"name": "m", "input_shape": [8, 8, 4], "channels": 1, "n_class": 3,
                    "blocks": 1, "filters": 4, "epochs": 1, "batch_size": 1}],
    })
    with pytest.raises(EngineError, match=r"class 2 \(label 7\)"):
        Engine(spec).fit()


def test_several_missing_classes_are_listed_together(tmp_path):
    root = dataset_with_foreground(tmp_path / "many", 0.05)
    spec = spec_for(root, [[0, 0, 0], [255, 255, 255], [128, 0, 0], [0, 128, 0]])
    with pytest.raises(EngineError) as raised:
        Engine(spec).fit()
    message = str(raised.value)
    assert "class 2" in message and "class 3" in message
    assert " and " in message          # listed as prose, not as a repr


def test_a_high_unmatched_fraction_warns_without_refusing(tmp_path):
    """Both classes present, but a third of the mask matches nothing. Legitimate causes
    exist - a stray annotation colour - so this is a warning."""
    root = tmp_path / "mixed"
    for index in range(6):
        sample = root / f"case_{index}"
        (sample / "images").mkdir(parents=True, exist_ok=True)
        (sample / "masks").mkdir(parents=True, exist_ok=True)
        mask = np.zeros((64, 64, 3), np.uint8)
        mask[5:25, 5:25] = (255, 255, 255)      # declared
        mask[30:50, 30:50] = (77, 88, 99)       # not declared, and large
        Image.fromarray(mask).save(sample / "masks" / "m.png")
        Image.fromarray(np.full((64, 64, 3), 30, np.uint8)).save(sample / "images" / "a.png")

    spec = spec_for(root, [[0, 0, 0], [255, 255, 255]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Engine(spec).fit()
    messages = [str(w.message) for w in caught]
    assert any("match no colormap entry" in m for m in messages), messages


def test_the_check_can_be_turned_off(tmp_path):
    root = dataset_with_foreground(tmp_path / "off", 0.05)
    spec = spec_for(root, [[0, 0, 0], [128, 0, 0]])
    Engine(spec, check_masks=False).fit()          # trains on nothing, as asked


def test_a_prediction_only_run_is_not_refused(tmp_path):
    """`fit = False` loads weights and trains nothing, so a refusal there would stop a
    run that never looks at a training mask. Proven with a wrong colormap: the same spec
    that `fit()` refuses above goes through when there is nothing to train.

    Not asserted by patching the check out - that would prove the test can patch. The
    weights are real, exported from a matching model, and the run completes.
    """
    from pyplatypus.models import build_model
    from pyplatypus.weights import export_weights

    root = dataset_with_foreground(tmp_path / "noop", 0.05)
    wrong = spec_for(root, [[0, 0, 0], [128, 0, 0]])
    weights = export_weights(build_model(wrong.models[0]), wrong.models[0],
                             tmp_path / "w.safetensors")

    loading = spec_for(root, [[0, 0, 0], [128, 0, 0]],
                       weights=str(weights), fit=False)
    engine = Engine(loading)
    engine.fit()
    assert engine.runs["m"].trained is False


# --- the sample has to cross the dataset, not its beginning ----------------------------

def test_a_class_only_in_the_last_samples_is_still_found(tmp_path):
    """The check reads a subset, and real datasets arrive sorted - by patient, by
    acquisition date, sometimes by class. A prefix would refuse this dataset, which is
    entirely correct: the only labelled tissue is in the last 3% of a sorted listing.
    """
    root = tmp_path / "sorted"
    total, positives = 200, 6
    for index in range(total):
        sample = root / f"case_{index:04d}"
        (sample / "images").mkdir(parents=True, exist_ok=True)
        (sample / "masks").mkdir(parents=True, exist_ok=True)
        mask = np.zeros((64, 64, 3), np.uint8)
        if index >= total - positives:
            mask[10:20, 10:20] = (255, 255, 255)
        Image.fromarray(mask).save(sample / "masks" / "m.png")
        Image.fromarray(np.full((64, 64, 3), 30, np.uint8)).save(sample / "images" / "a.png")

    spec = spec_for(root, [[0, 0, 0], [255, 255, 255]])
    dataset = Engine(spec, check_masks=False).dataset(spec.models[0], "train")

    # The prefix a naive check would have read carries nothing at all...
    prefix = sorted(p.name for p in root.iterdir())[:50]
    assert all(not name.endswith(tuple(f"{n:04d}" for n in range(total - positives, total)))
               for name in prefix)

    # ...and the visit order finds the class anyway, so fit() proceeds. It finds it almost
    # at once, because the order opens with both ends of the dataset.
    report = dataset.inspect_masks(limit=50)
    assert report.missing_classes == []
    assert report.total_samples == total
    Engine(spec).fit()


def test_the_report_says_how_much_of_the_dataset_it_looked_at(tmp_path):
    """Both numbers, because a refusal that says "checked 5 of 536" can be judged and one
    that says "the class is absent" cannot."""
    root = dataset_with_foreground(tmp_path / "span", 0.05, n=100)
    spec = spec_for(root, [[0, 0, 0], [255, 255, 255]])
    dataset = Engine(spec, check_masks=False).dataset(spec.models[0], "train")

    report = dataset.inspect_masks(limit=10)
    assert report.total_samples == 100
    assert 0 < report.samples_checked <= 10


# --- the scan stops early when there is nothing to find --------------------------------

def test_a_healthy_dataset_is_answered_in_a_few_reads(tmp_path):
    """Cost belongs where the doubt is. On Data Science Bowl, where every sample carries
    one mask file per nucleus, scanning fifty samples took 3.8 seconds and the first few
    answered the question; this is why it stops."""
    root = dataset_with_foreground(tmp_path / "healthy", 0.05, n=200)
    spec = spec_for(root, [[0, 0, 0], [255, 255, 255]])
    dataset = Engine(spec, check_masks=False).dataset(spec.models[0], "train")

    report = dataset.inspect_masks(limit=50)
    assert report.missing_classes == []
    assert report.samples_checked < 50
    # ...but not from one file, which would make the unmatched figure meaningless.
    assert report.samples_checked >= 5


def test_a_suspicious_dataset_is_scanned_to_the_limit(tmp_path):
    """The mirror image: while a class is still missing there is nothing to stop for."""
    root = dataset_with_foreground(tmp_path / "suspicious", 0.05, n=200)
    spec = spec_for(root, [[0, 0, 0], [128, 0, 0]])
    dataset = Engine(spec, check_masks=False).dataset(spec.models[0], "train")

    report = dataset.inspect_masks(limit=30)
    assert report.missing_classes == [1]
    assert report.samples_checked == 30


def test_any_prefix_of_the_visit_order_spans_the_dataset(tmp_path):
    """The property that makes an early exit safe. Without it, stopping early would read
    only the beginning - which is the bias the spread was there to avoid."""
    from pyplatypus.data.dataset import _spread

    order = _spread(10_000, 10)
    assert len(set(order)) == len(order)
    for length in (2, 3, 5):
        head = order[:length]
        assert min(head) < 1_000 and max(head) > 9_000, head

    # Degenerate shapes answer rather than raise.
    assert _spread(1, 5) == [0]
    assert _spread(0, 5) == []
    assert sorted(_spread(6, 50)) == list(range(6))
