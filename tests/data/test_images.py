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
