"""Writing a predicted volume back onto the scan it came from.

The test that earns its keep is `test_a_mask_lands_on_its_scan_when_the_file_is_not_canonical`.
Everything else here checks refusals; that one checks the claim the writer exists for, and it
is the only one that can tell the right affine from a plausible one. A file stored in LAS
reads back through `read_volume` reoriented, so a mask written with the file's **stored**
affine is reflected with respect to the scan - and opens in any viewer without complaint.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyplatypus.data.volumes import (
    VolumeError,
    read_volume,
    save_class_volumes,
    save_volumes,
)

nib = pytest.importorskip("nibabel")


def write_scan(path, array, affine=None):
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine), str(path))
    return path


def lesion(shape=(6, 8, 4)):
    """A label map with one blob off-centre, so a flip is visible in the array."""
    labels = np.zeros(shape, dtype=np.int16)
    labels[1:3, 1:4, 1:3] = 1
    return labels


def one_hot(labels, classes=2):
    return np.eye(classes, dtype=np.float32)[labels]


def test_a_mask_lands_on_its_scan_when_the_file_is_not_canonical(tmp_path):
    """The claim: a prediction written against a file is in that file's own frame.

    The scan is stored with its first axis reversed, so `read_volume` reorients it and a
    prediction is in canonical space. Reading the written mask back through the same reader
    has to give the array that was written - which is only true if the writer reoriented the
    reference the same way the reader did.
    """
    flipped = np.diag([-1.0, 1.0, 1.0, 1.0])
    labels = lesion()
    scan = write_scan(tmp_path / "ct.nii.gz", labels.astype(np.float32), affine=flipped)

    canonical = read_volume(scan, nearest=True)[..., 0].astype(np.int16)
    written = save_volumes(
        [one_hot(canonical)], tmp_path / "out", reference=[scan], cases=["case_01"]
    )

    back = read_volume(written[0], nearest=True)[..., 0].astype(np.int16)
    assert np.array_equal(back, canonical)
    # And the control that makes the assertion mean something: the file really is stored
    # the other way round, so a writer using the stored affine would differ here.
    assert not np.array_equal(canonical, labels)


def test_the_written_file_carries_the_references_canonical_affine(tmp_path):
    flipped = np.diag([-1.0, 1.0, 2.5, 1.0])
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32), affine=flipped)
    canonical = read_volume(scan, nearest=True)[..., 0].astype(np.int16)

    written = save_volumes([one_hot(canonical)], tmp_path / "out", reference=[scan])

    expected = nib.as_closest_canonical(nib.load(str(scan))).affine
    assert np.allclose(nib.load(str(written[0])).affine, expected)


def test_labels_are_integers_rather_than_probabilities(tmp_path):
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))
    probabilities = np.stack([np.full((6, 8, 4), 0.4), np.full((6, 8, 4), 0.6)], axis=-1)

    written = save_volumes([probabilities], tmp_path / "out", reference=[scan])

    image = nib.load(str(written[0]))
    assert image.get_data_dtype() == np.int16
    assert set(np.unique(np.asarray(image.dataobj))) == {1}


def test_an_overlapping_prediction_is_refused_rather_than_argmaxed(tmp_path):
    """A label map holds one integer per voxel, so `regions` go one file per class."""
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))
    overlapping = np.zeros((6, 8, 4, 2), dtype=np.float32)
    overlapping[..., 0] = 0.9
    overlapping[2, 2, 2, 1] = 0.8

    with pytest.raises(VolumeError, match="save_class_volumes"):
        save_volumes([overlapping], tmp_path / "out", reference=[scan])


def test_a_softmax_prediction_is_not_refused(tmp_path):
    """The control for the test above: without it, every ordinary run would start failing."""
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))
    softmax = np.full((6, 8, 4, 3), 1 / 3, dtype=np.float32)
    softmax[..., 1] = 0.4
    softmax[..., 0] = 0.27

    assert save_volumes([softmax], tmp_path / "out", reference=[scan])


def test_a_mask_of_the_wrong_shape_is_refused_with_both_shapes(tmp_path):
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))
    wrong = np.zeros((6, 8, 5, 2), dtype=np.float32)

    with pytest.raises(VolumeError, match="6 x 8 x 5.*6 x 8 x 4"):
        save_volumes([wrong], tmp_path / "out", reference=[scan])


def test_case_names_that_repeat_are_refused(tmp_path):
    """One directory per case, the same `ct.nii.gz` in each: the stems are not distinct.

    Without this the second mask overwrites the first, the returned list of paths still has
    two entries, and nothing says a mask was lost.
    """
    scans = [
        write_scan(tmp_path / "case_01" / "ct.nii.gz", lesion().astype(np.float32)),
        write_scan(tmp_path / "case_02" / "ct.nii.gz", lesion().astype(np.float32)),
    ]
    masks = [one_hot(lesion()), one_hot(lesion())]
    stems = [path.name.replace(".nii.gz", "") for path in scans]

    with pytest.raises(VolumeError, match="repeat"):
        save_volumes(masks, tmp_path / "out", reference=scans, cases=stems)

    written = save_volumes(masks, tmp_path / "out", reference=scans, cases=["case_01", "case_02"])
    assert len({path.name for path in written}) == 2


def test_a_reference_per_mask_is_required(tmp_path):
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))

    with pytest.raises(VolumeError, match="2 masks and 1 reference"):
        save_volumes([one_hot(lesion())] * 2, tmp_path / "out", reference=[scan])


def test_something_that_is_not_a_volume_is_refused(tmp_path):
    """Rank 2, because rank 3 is a label map now and rank 4 is channels."""
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))

    with pytest.raises(VolumeError, match="channels-last volumes"):
        save_volumes([np.zeros((8, 8), dtype=np.float32)], tmp_path / "out", reference=[scan])


def test_a_label_map_is_written_without_being_expanded_first(tmp_path):
    """`#251`: R's `predict()` returns class indices where this one returns probabilities.

    The same labels through both doors have to produce the same file, or the two packages
    write different answers from one prediction.
    """
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))
    labels = read_volume(scan, nearest=True)[..., 0].astype(np.int16)

    from_labels = save_volumes([labels], tmp_path / "a", reference=[scan], cases=["case"])
    from_channels = save_volumes(
        [one_hot(labels)], tmp_path / "b", reference=[scan], cases=["case"]
    )

    assert np.array_equal(
        np.asarray(nib.load(str(from_labels[0])).dataobj),
        np.asarray(nib.load(str(from_channels[0])).dataobj),
    )


def test_a_label_map_skips_the_overlap_refusal_because_it_cannot_overlap(tmp_path):
    """The refusal protects the collapse, and a label map is not collapsed.

    Two classes, written as indices, where the channel form of the same thing would be
    refused: an integer per voxel cannot say a voxel is in both, so there is nothing to
    detect and nothing lost.
    """
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))
    labels = read_volume(scan, nearest=True)[..., 0].astype(np.int16)

    written = save_volumes([labels], tmp_path / "out", reference=[scan])

    assert set(np.unique(np.asarray(nib.load(str(written[0])).dataobj))) == {0, 1}


def test_a_bare_rank_four_array_is_refused_rather_than_guessed(tmp_path):
    """A stack of label maps and one volume's channels are both rank 4.

    `save_class_masks` names the same ambiguity for pictures and the R package resolves it
    the same way, so a caller meets one rule rather than a third.
    """
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))

    with pytest.raises(VolumeError, match="could be either"):
        save_volumes(np.zeros((6, 8, 4, 2), dtype=np.float32), tmp_path / "out", reference=[scan])

    # And the way out that the message names: a list says which it is.
    assert save_volumes(
        [np.zeros((6, 8, 4, 2), dtype=np.float32)], tmp_path / "out", reference=[scan]
    )


def test_volumes_of_different_ranks_are_refused(tmp_path):
    scans = [
        write_scan(tmp_path / "a.nii.gz", lesion().astype(np.float32)),
        write_scan(tmp_path / "b.nii.gz", lesion().astype(np.float32)),
    ]
    mixed = [lesion(), one_hot(lesion())]

    with pytest.raises(VolumeError, match="different ranks"):
        save_volumes(mixed, tmp_path / "out", reference=scans)


def test_save_class_volumes_still_wants_channels(tmp_path):
    """One binary file per class is a question about channels, and a label map has none."""
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))

    with pytest.raises(VolumeError, match="channels-last volumes"):
        save_class_volumes([lesion()], tmp_path / "out", reference=[scan], classes=["core", "rim"])


def test_one_file_per_class_with_the_name_the_filesystem_can_hold(tmp_path):
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))
    overlapping = np.zeros((6, 8, 4, 2), dtype=np.float32)
    overlapping[1:3, 1:4, 1:3, :] = 0.9

    written = save_class_volumes(
        [overlapping],
        tmp_path / "out",
        reference=[scan],
        classes=["whole tumour", "enhancing"],
        cases=["case_01"],
    )

    assert [path.name for path in written] == [
        "case_01_whole_tumour.nii.gz",
        "case_01_enhancing.nii.gz",
    ]
    both = [np.asarray(nib.load(str(path)).dataobj) for path in written]
    assert np.array_equal(both[0], both[1])
    assert both[0].sum() == 2 * 3 * 2


def test_a_class_nothing_reached_is_written_empty_rather_than_skipped(tmp_path):
    """A missing file and an empty one are different findings, and only one is true."""
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))
    quiet = np.zeros((6, 8, 4, 2), dtype=np.float32)
    quiet[..., 0] = 0.9

    written = save_class_volumes(
        [quiet], tmp_path / "out", reference=[scan], classes=["core", "oedema"]
    )

    assert len(written) == 2
    assert np.asarray(nib.load(str(written[1])).dataobj).sum() == 0


def test_two_class_names_the_filesystem_cannot_keep_apart_are_refused(tmp_path):
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))
    prediction = np.zeros((6, 8, 4, 2), dtype=np.float32)

    with pytest.raises(VolumeError, match="same filename"):
        save_class_volumes(
            [prediction],
            tmp_path / "out",
            reference=[scan],
            classes=["whole tumour", "whole/tumour"],
        )


def test_a_class_name_per_channel_is_required(tmp_path):
    scan = write_scan(tmp_path / "ct.nii.gz", lesion().astype(np.float32))
    prediction = np.zeros((6, 8, 4, 3), dtype=np.float32)

    with pytest.raises(VolumeError, match="3 channels and 2 class names"):
        save_class_volumes(
            [prediction], tmp_path / "out", reference=[scan], classes=["core", "oedema"]
        )
