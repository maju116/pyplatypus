import numpy as np
import pytest

from pyplatypus.data import read_image, stitch, tile, to_float
from pyplatypus.data.images import ImageError


@pytest.mark.parametrize(
    "shape,splits",
    [
        ((256, 384, 3), (2, 3)),
        ((8, 8, 1), (4, 2)),
        ((64, 64, 64, 1), (2, 2, 2)),  # 3D, same code
    ],
)
def test_tile_and_stitch_are_exact_inverses(shape, splits):
    """This is the guarantee the old package never had: what was cut up comes back."""
    original = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    assert np.array_equal(stitch(tile(original, splits), splits), original)


def test_tile_count_and_shape():
    tiles = tile(np.zeros((256, 384, 3)), (2, 3))
    assert tiles.shape == (6, 128, 128, 3)


def test_tiles_come_out_in_row_major_order():
    """stitch depends on this order, so it is pinned down by a test."""
    grid = np.array([[0, 1], [2, 3]], dtype=np.float32)[..., None]
    tiles = tile(grid, (2, 2))
    assert [float(t.squeeze()) for t in tiles] == [0.0, 1.0, 2.0, 3.0]


def test_indivisible_shape_is_refused():
    with pytest.raises(ImageError, match="divide exactly"):
        tile(np.zeros((255, 256, 3)), (2, 2))


def test_stitch_checks_the_tile_count():
    with pytest.raises(ImageError, match="needs 6 tiles"):
        stitch(np.zeros((4, 8, 8, 3)), (2, 3))


def test_read_image_resizes_to_height_width(tmp_path):
    """PIL takes (width, height) and every shape here is (height, width). Getting that
    backwards is the classic silent bug, so it gets a test."""
    from tests.conftest import write_png

    write_png(tmp_path / "a.png", np.zeros((10, 20, 3)))
    assert read_image(tmp_path / "a.png", channels=3, size=(64, 32)).shape == (64, 32, 3)


def test_masks_are_resized_without_inventing_colours(tmp_path):
    from tests.conftest import write_png

    mask = np.zeros((10, 10, 3), np.uint8)
    mask[5:] = 255
    write_png(tmp_path / "m.png", mask)

    nearest = read_image(tmp_path / "m.png", channels=3, size=(33, 33), nearest=True)
    smooth = read_image(tmp_path / "m.png", channels=3, size=(33, 33), nearest=False)
    assert set(np.unique(nearest)) == {0, 255}
    assert len(np.unique(smooth)) > 2  # interpolation blends, which would break classes


def test_greyscale_gets_a_channel_axis(tmp_path):
    from tests.conftest import write_png

    write_png(tmp_path / "g.png", np.zeros((8, 8, 3)))
    assert read_image(tmp_path / "g.png", channels=1).shape == (8, 8, 1)


def test_to_float_scales_only_integers():
    assert to_float(np.array([[255]], np.uint8))[0, 0] == pytest.approx(1.0)
    assert to_float(np.array([[0.5]], np.float32))[0, 0] == pytest.approx(0.5)


# --- a label map's values, which 8-bit greyscale cannot hold -----------------------------
#
# `pyplatypus#346`. `_PIL_MODE[1]` is `"L"`, so `convert("L")` flattened every value above
# 255 to 255 and `np.asarray(..., uint8)` would have clipped whatever survived. A label
# map's values are class identities rather than brightness, so the clip merged two classes
# into one - at an unmatched share of exactly zero, which is the one number the mask check
# reads.


def _write_label_map(path, array):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_a_16_bit_label_map_keeps_the_values_above_255(tmp_path):
    values = np.array([[0, 1, 2, 3], [255, 256, 300, 1000]], dtype=np.uint16)
    _write_label_map(tmp_path / "m.png", values)

    read = read_image(tmp_path / "m.png", channels=1, nearest=True, label_map=True)
    assert read[..., 0].tolist() == values.tolist()

    # The reading that made this a defect rather than a limitation of the format.
    clipped = read_image(tmp_path / "m.png", channels=1, nearest=True)
    assert clipped[..., 0].tolist() == [[0, 1, 2, 3], [255, 255, 255, 255]]


def test_a_32_bit_label_map_keeps_its_values_too(tmp_path):
    """A 32-bit TIFF, which is the other mode a label map arrives in."""
    values = np.array([[0, 7, 70000]], dtype=np.int32)
    _write_label_map(tmp_path / "m.tif", values)
    read = read_image(tmp_path / "m.tif", channels=1, nearest=True, label_map=True)
    assert read[..., 0].tolist() == values.tolist()


def test_values_below_256_read_the_same_either_way(tmp_path):
    """The control that the clip was what moved them, rather than the depth.

    An 8-bit label map is the ordinary case and must not change, and a 16-bit file whose
    values all fit in a byte read correctly before this too.
    """
    values = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint16)
    _write_label_map(tmp_path / "wide.png", values)
    byte = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint8)
    _write_label_map(tmp_path / "byte.png", byte)

    for name in ("wide.png", "byte.png"):
        plain = read_image(tmp_path / name, channels=1, nearest=True)
        widened = read_image(tmp_path / name, channels=1, nearest=True, label_map=True)
        assert plain[..., 0].tolist() == values.tolist()
        assert widened[..., 0].tolist() == values.tolist()


def test_an_8_bit_label_map_stored_as_rgb_still_reads_its_values(tmp_path):
    """The case a mask-only change to `"I"` must not break: `convert` reads a three-channel
    file's luminance either way, and a grey RGB mask is a label map somebody saved wide."""
    from tests.conftest import write_png

    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[0, 0] = (1, 1, 1)
    rgb[1, 1] = (2, 2, 2)
    write_png(tmp_path / "rgb.png", rgb)

    read = read_image(tmp_path / "rgb.png", channels=1, nearest=True, label_map=True)
    assert read[..., 0].tolist() == [[1, 0], [0, 2]]


def test_a_resize_of_a_label_map_keeps_the_values(tmp_path):
    """Nearest, which is what every mask route asks for - the widening must survive it."""
    values = np.array([[0, 1000], [1000, 0]], dtype=np.uint16)
    _write_label_map(tmp_path / "m.png", values)
    read = read_image(tmp_path / "m.png", channels=1, size=(4, 4), nearest=True, label_map=True)
    assert set(np.unique(read).tolist()) == {0, 1000}


def test_a_label_map_with_several_channels_is_refused(tmp_path):
    """Because a label map has one, and a three-channel mask is a picture with a colormap.

    The refusal rather than a silent luminance, so that a caller asking for the wrong thing
    hears about it instead of getting a plausible array.
    """
    from tests.conftest import write_png

    write_png(tmp_path / "p.png", np.zeros((4, 4, 3)))
    with pytest.raises(ImageError, match="label map has one channel"):
        read_image(tmp_path / "p.png", channels=3, label_map=True)


def _voc_like_palette() -> list[int]:
    """A palette whose colours are nothing like the indices, which is the whole point.

    A palette where entry `k` happens to be grey `(k, k, k)` would make a luminance read
    agree with an index read, and then nothing below could fail.
    """
    palette: list[int] = []
    for index in range(256):
        palette += [(index * 37) % 256, (index * 17) % 256, (index * 97) % 256]
    return palette


def _write_palettised(path, indices):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = Image.fromarray(indices, mode="P")
    handle.putpalette(_voc_like_palette())
    handle.save(path)
    return handle


def test_a_palettised_label_map_reads_its_indices_rather_than_its_palette(tmp_path):
    """A mode-`P` mask stores one index per pixel and the index IS the class.

    Pascal VOC's segmentation masks are stored this way and `rplatypus` ships that palette
    as `voc_colormap`, so this is the format a reader meets rather than a corner of PIL.
    `convert` expands the palette before widening anything, so `#346`'s fix did not reach
    it: indices 0, 1, 2, 15 came back as 0, 32, 64, 182 against the palette below.
    `pyplatypus#353`.
    """
    indices = np.array([[0, 1, 2, 15], [20, 255, 0, 1]], dtype=np.uint8)
    _write_palettised(tmp_path / "p.png", indices)

    read = read_image(tmp_path / "p.png", channels=1, nearest=True, label_map=True)
    assert read[..., 0].tolist() == indices.tolist()


def test_the_colormap_route_still_expands_a_palette(tmp_path):
    """The control for the test above, and the thing its fix must not break.

    A `colormap` user's file is a picture of classes and the palette is how it says which
    is which, so that route has to keep returning colours. Asserted per pixel rather than
    on the first one, which is index 0 and black in most palettes - a reading that agrees
    for the wrong reason.
    """
    indices = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    _write_palettised(tmp_path / "p.png", indices)
    palette = _voc_like_palette()

    rgb = read_image(tmp_path / "p.png", channels=3, nearest=True)
    for row in range(2):
        for column in range(2):
            index = int(indices[row, column])
            assert rgb[row, column].tolist() == palette[3 * index : 3 * index + 3]


def test_a_bilevel_label_map_reads_0_and_1_rather_than_0_and_255(tmp_path):
    """One bit per pixel, which `convert` scales to the full range.

    A binary mask written as a 1-bit PNG - which is what several tools do to save space -
    stores 0 and 1 and arrived as 0 and 255, so `labels: [0, 1]` read class 1 as absent on
    every row. Found while scoping `#353`'s fix rather than in the report: the mode was
    mangled before `#346` too, by `convert("L")`, so it is the same defect on a mode
    nobody had looked at.
    """
    from PIL import Image

    values = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    Image.fromarray(values * 255, mode="L").convert("1").save(tmp_path / "b.png")
    assert Image.open(tmp_path / "b.png").mode == "1", "the fixture is not bilevel"

    read = read_image(tmp_path / "b.png", channels=1, nearest=True, label_map=True)
    assert read[..., 0].tolist() == values.tolist()


def test_a_float_label_map_is_rounded_rather_than_truncated(tmp_path):
    """`pyplatypus#361`: `convert("I")` truncates towards zero.

    `label_map`'s own docstring promises that *"a 16-bit PNG or a 32-bit TIFF keeps the
    values it stores"*, and a float32 TIFF is a 32-bit TIFF. It kept them by truncation, so
    a class written as 0.9999 by whatever produced the file became 0 - one class down, with
    every voxel still matching something and the mask check reporting **0.0 unmatched**.
    That is `#346`'s silent costume on the mode its fix did not reach.

    `labels_to_classes` compares to within half a unit for exactly this, and never got the
    chance: the truncation happens in the reader.
    """
    from PIL import Image

    values = np.array([[0.0, 0.9999, 1.9999, 2.0001]], dtype=np.float32)
    Image.fromarray(values, mode="F").save(tmp_path / "f.tif")
    assert Image.open(tmp_path / "f.tif").mode == "F", "the fixture is not a float TIFF"

    read = read_image(tmp_path / "f.tif", channels=1, nearest=True, label_map=True)
    assert read[..., 0].tolist() == [[0, 1, 2, 2]]

    # THE CONTROL HAS TO HOLD THE INTENDED CLASSES. Writing `values.astype(np.int32)`
    # truncates before saving, so the file would already hold [0, 0, 1, 2] and would read
    # back "correctly" for the wrong reason - which is what the first version of this
    # probe did.
    Image.fromarray(np.array([[0, 1, 2, 2]], dtype=np.int32), mode="I").save(tmp_path / "i.tif")
    control = read_image(tmp_path / "i.tif", channels=1, nearest=True, label_map=True)
    assert control[..., 0].tolist() == [[0, 1, 2, 2]]


def test_the_two_readings_of_one_float_label_map_agree(tmp_path):
    """The reader's answer and the tolerance's answer, on the same four values.

    A volume carries floats all the way to `labels_to_classes`, which resolves them by its
    half-unit tolerance; a float TIFF is resolved in the reader. Those are two rules for one
    question, so the thing worth asserting is that they agree - which is the whole argument
    for rounding rather than reading the file as float and widening the dtype.
    """
    from PIL import Image

    from pyplatypus.data.masks import labels_to_classes

    values = np.array([[0.0, 0.9999, 1.9999, 2.0001]], dtype=np.float32)
    Image.fromarray(values, mode="F").save(tmp_path / "f.tif")

    read = read_image(tmp_path / "f.tif", channels=1, nearest=True, label_map=True)
    through_the_reader, _ = labels_to_classes(read, [0, 1, 2])
    through_the_tolerance, _ = labels_to_classes(values.reshape(1, -1, 1), [0, 1, 2])

    assert through_the_reader.ravel().tolist() == through_the_tolerance.ravel().tolist()
