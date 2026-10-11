"""Reading per-object masks, and the three things that make a case unreadable.

The Data Science Bowl's layout is the one this reads: a case directory with `masks/` holding
one PNG per nucleus. What matters here is that the read is ordered, that an unusable case says
so, and that nothing silently becomes an object or stops being one.

Everything here builds its own files. A test reading the real dataset would skip on every
machine that does not hold 670 case directories, including CI, and a skip reads exactly like a
pass - `examples/score_dsbowl_instances.py` is what exercises the real layout, and it produced
the baseline number rather than an assertion.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyplatypus.instances import InstanceError, read_instance_masks, read_instances


def write_case(root, objects, name="case"):
    """A case directory with one PNG per object, each a rectangle given as a slice pair."""
    folder = root / name / "masks"
    folder.mkdir(parents=True)
    for index, (rows, columns) in enumerate(objects):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[rows, columns] = 255
        Image.fromarray(mask).save(folder / f"{index:02d}.png")
    return root / name


def test_every_object_comes_back_separately(tmp_path):
    case = write_case(tmp_path, [(slice(0, 2), slice(0, 2)), (slice(5, 7), slice(5, 7))])

    masks = read_instance_masks(case)
    assert len(masks) == 2
    assert all(mask.dtype == bool for mask in masks)
    assert [int(mask.sum()) for mask in masks] == [4, 4]

    labels = read_instances(case)
    assert sorted(np.unique(labels)) == [0, 1, 2]


def test_the_read_is_sorted_by_filename(tmp_path):
    """Not load-bearing for the metric, which compares every pair - but an unordered read
    makes two runs' per-case output impossible to diff, and that is the first thing anybody
    does when a number moves."""
    case = tmp_path / "case" / "masks"
    case.mkdir(parents=True)
    for name, column in (("zz.png", 0), ("aa.png", 7)):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:, column] = 255
        Image.fromarray(mask).save(case / name)

    masks = read_instance_masks(tmp_path / "case")
    # `aa.png` marks column 7, so if the read is sorted it is the first mask.
    assert masks[0][0, 7] and not masks[0][0, 0]


def test_a_case_with_no_masks_directory_is_refused(tmp_path):
    (tmp_path / "case").mkdir()
    with pytest.raises(InstanceError, match="no masks/ directory"):
        read_instance_masks(tmp_path / "case")


def test_a_masks_directory_with_no_files_is_refused(tmp_path):
    (tmp_path / "case" / "masks").mkdir(parents=True)
    with pytest.raises(InstanceError, match="holds no .png files"):
        read_instance_masks(tmp_path / "case")


def test_an_empty_mask_file_is_refused(tmp_path):
    """An object covering no pixel would be counted as a truth object nothing can match, so
    it would subtract from the score without ever having been on the image."""
    case = tmp_path / "case" / "masks"
    case.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(case / "empty.png")

    with pytest.raises(InstanceError, match="is empty"):
        read_instance_masks(tmp_path / "case")


def test_overlapping_files_are_refused_when_they_become_a_label_map(tmp_path):
    """Reading them is fine; collapsing them to one integer per pixel is where the question
    is, and it is the open one (pyplatypus#158)."""
    case = write_case(tmp_path, [(slice(0, 4), slice(0, 4)), (slice(2, 6), slice(2, 6))])

    assert len(read_instance_masks(case)) == 2
    with pytest.raises(InstanceError, match="overlaps an earlier one"):
        read_instances(case)
