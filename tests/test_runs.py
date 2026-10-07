"""What a run leaves behind, beyond the weights.

The weights are not the whole of a trained model. A segmentation model is meaningless
without the colormap that says what its channels are; a detector is meaningless without the
anchors its boxes are relative to - and when those were fitted rather than named, the run is
the only place they exist. Both were recoverable only by remembering to export weights.

The assertion that matters is not that a file appears. It is that **the record runs again**:
`from_dict(record["specification"])` has to give back a specification, and for a detector the
anchors have to be there, because the specification that produced it does not name them.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from pyplatypus import Engine, build_engine, from_dict, read_record


def tiny_segmentation(root, output_dir=None):
    for split in ("train", "valid"):
        for n in range(2):
            sample = root / split / f"s{n}"
            (sample / "images").mkdir(parents=True, exist_ok=True)
            (sample / "masks").mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((32, 32, 3), 10, np.uint8)).save(sample / "images" / "i.png")
            mask = np.zeros((32, 32, 3), np.uint8)
            mask[8:20, 8:20] = 255
            Image.fromarray(mask).save(sample / "masks" / "m.png")

    config = {
        "task": "semantic_segmentation",
        "data": {
            "train_path": str(root / "train"),
            "validation_path": str(root / "valid"),
            "colormap": [[0, 0, 0], [255, 255, 255]],
        },
        "models": [
            {
                "name": "u",
                "input_shape": [32, 32],
                "blocks": 2,
                "filters": 4,
                "epochs": 1,
                "batch_size": 2,
            }
        ],
    }
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    return from_dict(config)


# --- only when asked for -----------------------------------------------------------------


def test_nothing_is_written_when_output_dir_was_not_given(tmp_path):
    """A field with a default cannot otherwise be told from one the user set to that
    default, and writing files into somebody's working directory because a default exists
    is not a thing to do by surprise."""
    spec = tiny_segmentation(tmp_path)
    Engine(spec, device="cpu", check_masks=False).fit()

    assert spec.output_dir == "platypus_output"
    assert not (tmp_path / "platypus_output").exists()
    assert not (tmp_path.parent / "platypus_output").exists()


def test_a_record_appears_when_it_was(tmp_path):
    out = tmp_path / "runs"
    spec = tiny_segmentation(tmp_path, output_dir=out)
    Engine(spec, device="cpu", check_masks=False).fit()

    assert (out / "u" / "run.json").is_file()


def test_setting_output_dir_to_its_own_default_still_counts_as_asking(tmp_path, monkeypatch):
    """`model_fields_set` answers "did the user write this", not "is it different from the
    default" - which is the honest question, and the one `missing()` in R gets wrong once
    anything has assigned to the argument."""
    monkeypatch.chdir(tmp_path)
    spec = tiny_segmentation(tmp_path, output_dir="platypus_output")
    Engine(spec, device="cpu", check_masks=False).fit()

    assert (tmp_path / "platypus_output" / "u" / "run.json").is_file()


# --- what is in it -------------------------------------------------------------------------


def test_a_segmentation_record_runs_again(tmp_path):
    """The assertion that matters. A record that cannot be turned back into a
    specification is a summary, and a summary is for reading rather than for running."""
    out = tmp_path / "runs"
    Engine(tiny_segmentation(tmp_path, output_dir=out), device="cpu", check_masks=False).fit()

    record = read_record(out / "u" / "run.json")
    again = from_dict(record["specification"])

    assert again.data.colormap == [(0, 0, 0), (255, 255, 255)]
    assert tuple(again.models[0].input_shape) == (32, 32)
    assert record["model"] == "u"
    assert record["pyplatypus"]
    assert len(record["history"]) == 1


def test_a_segmentation_record_derives_nothing_and_says_so(tmp_path):
    """Honest rather than padded: the specification already carries the colormap, the
    input shape and the window, so there is nothing a segmentation run works out that it
    does not already say."""
    out = tmp_path / "runs"
    Engine(tiny_segmentation(tmp_path, output_dir=out), device="cpu", check_masks=False).fit()
    assert read_record(out / "u" / "run.json")["derived"] == {}


def test_a_detection_record_carries_the_fitted_anchors(tmp_path, detection_config, detection_root):
    """The reason this exists. The specification that produced this detector does not name
    its anchors, the same weights read with any others decode every box scaled by a fixed
    factor, and without the record the only copy is a weights sidecar - and only if
    somebody remembered to export."""
    out = tmp_path / "runs"
    detection_config["output_dir"] = str(out)
    engine = build_engine(from_dict(detection_config), device="cpu")
    engine.fit()

    record = read_record(out / "d" / "run.json")
    assert record["specification"]["models"][0]["anchors"] is None, (
        "the specification does not name them, which is why `derived` must"
    )
    assert record["derived"]["anchors_were_fitted"] is True
    assert np.asarray(record["derived"]["anchors"]).shape == (3, 2, 2)

    # The anchors in the record are the anchors the run used, to the digit.
    assert record["derived"]["anchors"] == [
        [list(pair) for pair in group] for group in engine.runs["d"].anchors
    ]
    assert record["derived"]["targets"]["placed"] > 0


def test_the_detection_record_re_runs_the_same_detector(tmp_path, detection_config, detection_root):
    """Specification plus `derived.anchors` is the pair that reproduces a detector. Put
    them together and the engine uses exactly those anchors rather than fitting new ones."""
    out = tmp_path / "runs"
    detection_config["output_dir"] = str(out)
    first = build_engine(from_dict(detection_config), device="cpu")
    first.fit()

    record = read_record(out / "d" / "run.json")
    config = record["specification"]
    config["models"][0]["anchors"] = record["derived"]["anchors"]
    config["models"][0]["anchors_per_grid"] = record["derived"]["anchors_per_grid"]
    config.pop("output_dir", None)

    second = build_engine(from_dict(config), device="cpu")
    second.fit()

    assert second.runs["d"].anchor_fit is None, "nothing should have been refitted"
    assert second.runs["d"].anchors == first.runs["d"].anchors


def test_a_record_given_anchors_says_they_were_not_fitted(tmp_path, detection_config):
    out = tmp_path / "runs"
    given = [
        [[0.30, 0.30], [0.20, 0.20]],
        [[0.16, 0.16], [0.12, 0.12]],
        [[0.08, 0.08], [0.05, 0.05]],
    ]
    detection_config["output_dir"] = str(out)
    detection_config["models"][0]["anchors"] = given
    build_engine(from_dict(detection_config), device="cpu").fit()

    derived = read_record(out / "d" / "run.json")["derived"]
    assert derived["anchors_were_fitted"] is False
    assert derived["anchors"] == given
    assert "anchor_mean_iou" not in derived


def test_the_record_is_plain_json(tmp_path, detection_config):
    """It crosses into R and into anything else, so nothing in it may need Python to read.
    numpy scalars are what a run produces plenty of and what json refuses."""
    out = tmp_path / "runs"
    detection_config["output_dir"] = str(out)
    build_engine(from_dict(detection_config), device="cpu").fit()

    text = (out / "d" / "run.json").read_text()
    json.loads(text)
    assert "array(" not in text
    assert "np." not in text
