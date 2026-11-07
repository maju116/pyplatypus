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

import dataclasses
import json

import numpy as np
import pytest
from PIL import Image

from pyplatypus import Engine, build_engine, from_dict, read_record
from pyplatypus.runs import write_record
from pyplatypus.training.trainer import History


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


def test_a_record_holding_a_NAN_is_written_as_null_so_R_can_read_it(tmp_path):
    """`pyplatypus#373`. A history legitimately holds a NaN - `dice` with `smooth: 0` on a
    class no image in the batch contains is 0/0, and `terminate_on_nan` exists to watch for
    it - and `json.dumps` writes the bare token `NaN` for it by extension. Python reads that
    back, which is why nothing here noticed; `jsonlite::fromJSON` refuses it outright, so
    the record of the one run anybody wants to inspect was the one R could not open.
    """
    spec = tiny_segmentation(tmp_path / "data", output_dir=tmp_path / "runs")
    path = write_record(
        spec,
        "m",
        history=[
            {"epoch": 1, "val_dice": float("nan"), "train_loss": 0.5},
            {"epoch": 2, "val_dice": np.float32("nan"), "train_loss": float("inf")},
        ],
    )

    text = path.read_text()
    # The token itself, which is what R's parser stops on. Anchored, because `NaN` is a
    # substring of nothing else here but a bare search would also match a key called it.
    assert ": NaN" not in text
    assert ": Infinity" not in text and ": -Infinity" not in text
    # What replaced it, written out: the specification itself carries nulls for every
    # field nobody set, so the count is taken on the two keys this test is about.
    assert text.count('"val_dice": null') == 2
    assert '"train_loss": null' in text

    back = read_record(path)
    assert back["history"][0]["val_dice"] is None
    assert back["history"][1]["val_dice"] is None
    assert back["history"][1]["train_loss"] is None
    # The finite neighbour is untouched: a sanitiser that flattened every number would
    # pass the assertions above and destroy the record.
    assert back["history"][0]["train_loss"] == 0.5
    assert back["history"][0]["epoch"] == 1


# --- which epoch the weights came from, which the record did not say ----------------------
#
# `pyplatypus#417`, the remainder of `#403`. The provenance is recorded where it is decided
# and read by all four score tables, and until this it **left the process with the session**:
# `write_record` reached past the history into `history.records`, so `run.json` said which
# epoch was best by every metric the run logged and not which epoch's weights it shipped.
# `History.to_dict()` carried the field since `#398` and had no caller at all.
#
# The column answers the question for somebody reading a table in the session that produced
# it. The record answers it for somebody holding a directory six months later, which is the
# case `trained_with` on a weights sidecar exists for - and a run whose weights were published
# had its epoch in neither place.
#
# It is a SIBLING of `history` rather than a field inside it, because that key is a list of
# per-epoch rows and R reads exactly it: `read_record()` turns it into a data frame, so a
# scalar in there would change the type for the one reader that exists.


def _with_early_stopping(root, output_dir, *, epochs, patience):
    """A tiny run that selects an epoch, so the provenance has a value to carry.

    `restore_best` is what makes the weights an epoch's rather than the last one's, and
    `monitor` is left at its default because the point here is the record rather than the
    monitor - `#398` is where the two are told apart.
    """
    spec = tiny_segmentation(root, output_dir=output_dir)
    config = spec.to_dict()
    config["models"][0]["epochs"] = epochs
    config["models"][0]["callbacks"] = [
        {"name": "early_stopping", "patience": patience, "restore_best": True}
    ]
    return from_dict(config)


def test_the_record_says_which_epoch_the_weights_came_from(tmp_path):
    """End to end, and compared against the TABLE rather than against a number.

    Built from a history the trainer produced, which the issue asks for in those words: a
    test that hands `write_record` a literal asserts that `json.dumps` works. So a real fit
    runs, `early_stopping(restore_best = TRUE)` decides which epoch the weights are from, and
    the record is then held to what `evaluate()` reports for the same run. Two readers, one
    definition - a record that answered differently from the table beside it would be worse
    than one that said nothing.
    """
    out = tmp_path / "runs"
    spec = _with_early_stopping(tmp_path, out, epochs=3, patience=5)
    built = Engine(spec, device="cpu", check_masks=False)
    built.fit()
    table = built.evaluate("validation")

    record = read_record(out / "u" / "run.json")
    assert "weights_epoch" in record, (
        "the record does not say which epoch the weights came from - which is the one fact "
        "about a run that cannot be worked out again from the directory it left behind"
    )
    # `evaluate` answers with a list of rows, and the column is read by its name. NOT through
    # a `hasattr` fallback, which is what the first version of this did: it answered `None`
    # when the accessor was wrong and reported the record disagreeing with the table by 3
    # against None - §4's probe giving the comfortable answer, here as a finding that was not
    # there. A missing column raises instead.
    assert len(table) == 1, f"one model, so one row - got {len(table)}"
    reported = table[0]["best_epoch"]
    assert record["weights_epoch"] == reported, (
        f"the record says epoch {record['weights_epoch']} and the score table says "
        f"{reported}. Both read `history.weights_epoch`, so this is two copies of one fact "
        "disagreeing rather than either of them being wrong on its own"
    )
    assert isinstance(record["weights_epoch"], int), (
        f"the epoch arrived as {type(record['weights_epoch']).__name__}; `restore_best` ran, "
        "so a run that selected an epoch should report the number and not null"
    )
    # The rows are untouched, which is the whole reason this is a sibling key: R turns this
    # one into a data frame and a scalar in it would change its type.
    assert isinstance(record["history"], list)
    assert len(record["history"]) == 3
    assert all(isinstance(row, dict) for row in record["history"])


def test_the_record_carries_a_NULL_epoch_when_the_weights_are_no_single_epochs(tmp_path):
    """The deliberate half: present and null, not absent.

    A run with no callback that replaces weights ended on its last epoch, which `history`
    already says by its length - so `None` is the honest answer rather than a gap, and a key
    that vanished would be indistinguishable from this release never having happened.
    """
    out = tmp_path / "runs"
    Engine(tiny_segmentation(tmp_path, output_dir=out), device="cpu", check_masks=False).fit()

    record = read_record(out / "u" / "run.json")
    assert "weights_epoch" in record
    assert record["weights_epoch"] is None
    # The control that this run really had no selection to report, so the null above is the
    # answer rather than a run that failed to record one.
    assert len(record["history"]) == 1


def test_every_run_LEVEL_FACT_a_history_carries_reaches_the_record(tmp_path):
    """The class rather than the two instances of it, read off `History` by introspection.

    `weights_epoch` was the field this release is about and `stop_reason` was missing by
    exactly the same mechanism, in the same object, beside it - and `spec/components.py` tells
    a reader that *"the run's `stop_reason` names the clock, so a run that ran out of time
    does not read as a run that converged"*, which was true in a session and false of the
    file. Adding one and leaving the other is §3F's omission with a release's delay.

    So nothing here names the fields. `write_record` reads them off `History.to_dict()` and
    this reads them off the dataclass, so a field added to `History` either reaches the
    record or fails this test - which is what closes the class instead of the instance.

    Each is given a value no default could produce, because a field dropped on the way would
    otherwise be indistinguishable from one that is legitimately empty.
    """
    carried = [field.name for field in dataclasses.fields(History) if field.name != "records"]
    assert carried, (
        "`History` declares nothing beside its rows, so this test is comparing nothing - "
        "the dataclass was restructured and this guard has to be re-aimed"
    )

    history = History(records=[{"epoch": 1, "train_loss": 0.5}])
    distinctive = {}
    for name in carried:
        value = {"stop_reason": "a reason nothing defaults to", "weights_epoch": 7}.get(name)
        assert value is not None, (
            f"`History` gained the field `{name}` and this test has no distinctive value for "
            "it. Add one here rather than relaxing the comparison: a field set to its own "
            "default cannot be told from a field that was dropped"
        )
        setattr(history, name, value)
        distinctive[name] = value

    spec = tiny_segmentation(tmp_path, output_dir=tmp_path / "runs")
    record = read_record(write_record(spec, "m", history=history))

    for name, value in distinctive.items():
        assert name in record, (
            f"a history carries `{name}` and the record does not. `write_record` reads them "
            "off `History.to_dict()`, so either that method stopped offering it or the "
            "record stopped writing what it offers"
        )
        assert record[name] == value, f"`{name}` arrived as {record[name]!r}, not {value!r}"
    # The rows still went where they always went, under their own key and as a list.
    assert record["history"] == [{"epoch": 1, "train_loss": 0.5}]


def test_a_history_field_that_would_overwrite_the_record_is_REFUSED(tmp_path):
    """The guard on the mechanism above, because reading fields off an object is open-ended.

    A `model` field on a history would otherwise rewrite the model's name and the record would
    say the wrong thing quietly, which is worse than a record that could not be written. The
    refusal names the key, so the repair is obvious from the message.
    """

    class _Colliding(History):
        def to_dict(self):
            return {"records": [], "model": "not the model's name", "written": "nor this"}

    spec = tiny_segmentation(tmp_path, output_dir=tmp_path / "runs")
    with pytest.raises(ValueError, match=r"\['model', 'written'\]"):
        write_record(spec, "m", history=_Colliding())

    # And the ordinary case is not refused, which is the control: a guard that fired on
    # everything would pass the assertion above and break every run.
    assert write_record(spec, "m", history=History(records=[{"epoch": 1}])).is_file()


#: The record's structural keys - everything in it that is not a history's. Written out by
#: hand on purpose, because this IS the file format: R's `test-records.R` asserts
#: `names(record)` with `expect_setequal`, so a key added here turns that suite red when the
#: pin moves, and a key added by ACCIDENT should fail in the engine first.
#:
#: It exists because a mutation survived without it. `_history_parts` popping the rows out of
#: `to_dict()` rather than reading them with `get` is the difference between a record with one
#: `history` key and a record carrying a duplicate `records` beside it - semantically the same
#: file, and a silent change to a published format that only R's suite could have caught.
#: §3F's own argument: the engine's answer has to be held in the engine's suite, or the next
#: person to read the R failure fixes the R side.
RECORD_KEYS = frozenset({"written", "pyplatypus", "model", "specification", "derived", "history"})


def test_the_record_holds_EXACTLY_the_keys_the_format_declares(tmp_path):
    """The format, from its two sources rather than from one list.

    The structural keys are this module's and the rest are whatever a `History` carries, so
    the expected set is built the same way the record is - and both an extra key and a missing
    one fail here. That is the assertion R has had since before this release and the engine
    had not.
    """
    out = tmp_path / "runs"
    spec = _with_early_stopping(tmp_path, out, epochs=2, patience=5)
    Engine(spec, device="cpu", check_masks=False).fit()
    record = read_record(out / "u" / "run.json")

    beside_the_rows = {
        field.name for field in dataclasses.fields(History) if field.name != "records"
    }
    expected = set(RECORD_KEYS) | beside_the_rows
    assert set(record) == expected, (
        f"the record's keys have changed.\n  only in the file: {sorted(set(record) - expected)}"
        f"\n  only expected:    {sorted(expected - set(record))}\n"
        "`run.json` is a published format and R pins this set exactly, so a key arriving or "
        "leaving is a decision - take it here, in `RECORD_KEYS` or on `History`, and move R's "
        "assertion with the pin"
    )
    # The control that the comparison is over something: an empty set equals an empty set,
    # and `History` having no fields would make the half above vacuous.
    assert beside_the_rows, "`History` declares nothing beside its rows - re-aim this guard"
    assert len(expected) > len(RECORD_KEYS)
