"""What `Image.open` is allowed to dispatch to, and why a name is not a guard.

PIL identifies a picture by its contents. Every reader here was handed a path and nothing
else, so every decoder PIL ships was reachable from any path a specification named - the EPS
plugin among them, which hands the file to Ghostscript. `READABLE_FORMATS` is the repair, and
these tests are about the half of it that is easy to get wrong: the refusal has to fire on
*contents*, which means a correctly named file with the wrong bytes is refused and a wrongly
named file with the right bytes is read.

Every refusal here carries the control that makes it mean something: the same file, opened
with its own format allowed, is a picture PIL reads happily. Without that control a refusal
is indistinguishable from a broken fixture, which is the shape `JOURNAL.md` keeps recording.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyplatypus.data.images import READABLE_FORMATS, ImageError, read_image, spatial_shape
from pyplatypus.instances.metrics import InstanceError
from pyplatypus.instances.read import read_instance_masks

#: A format no reader here has ever been pointed at, and one PIL can also WRITE, so the
#: fixture is a real picture of a real excluded format rather than a handful of bad bytes.
#: EPS would be the sharper example - it is the Ghostscript path - but writing one and then
#: proving it readable needs Ghostscript installed, so the control could not be run.
EXCLUDED = "XBM"


def write(path: Path, fmt: str) -> Path:
    """One small picture, written in `fmt`, under whatever name `path` carries."""
    mode = "1" if fmt == "XBM" else "L"
    Image.fromarray(np.zeros((8, 8), np.uint8), mode="L").convert(mode).save(path, format=fmt)
    return path


def test_the_excluded_format_is_refused_under_a_pictures_name(tmp_path):
    """A file whose bytes are an excluded format, named `.png`, is refused by both readers."""
    planted = write(tmp_path / "scan.png", EXCLUDED)

    # The control first, because it is what makes the two refusals below evidence: these
    # bytes ARE a picture, and PIL reads them the moment its format is allowed.
    with Image.open(planted, formats=[EXCLUDED]) as handle:
        assert handle.size == (8, 8)
    assert EXCLUDED not in READABLE_FORMATS

    with pytest.raises(ImageError) as read:
        read_image(planted, channels=1)
    with pytest.raises(ImageError) as shape:
        spatial_shape(planted)

    # The message has to say that the name was not what decided, because the extension is
    # the first thing anybody checks and here it was already right.
    for raised in (read, shape):
        assert "not what decides" in str(raised.value)
        for allowed in READABLE_FORMATS:
            assert allowed in str(raised.value)


@pytest.mark.parametrize("fmt", READABLE_FORMATS)
def test_every_allowed_format_reads_whatever_it_is_named(tmp_path, fmt):
    """The other direction: contents decide, so an allowed format reads under a foreign name.

    Parametrised over the list itself rather than over a few names, so a format added to
    `READABLE_FORMATS` is covered by arriving - and one added that PIL cannot actually read
    back fails here instead of at a user's first picture.
    """
    misnamed = write(tmp_path / f"case.{'tif' if fmt == 'PNG' else 'png'}", fmt)

    array = read_image(misnamed, channels=1)

    assert array.shape == (8, 8, 1)
    assert spatial_shape(misnamed) == (8, 8)


def test_the_instance_reader_refuses_the_same_plant(tmp_path):
    """The per-object reader globs `*.png` by name; the guard it needed is about contents."""
    case = tmp_path / "case_00"
    (case / "masks").mkdir(parents=True)
    good = write(case / "masks" / "00.png", "PNG")
    Image.open(good).close()  # the fixture is a PNG, which the glob cannot tell us
    one = np.zeros((8, 8), np.uint8)
    one[2:6, 2:6] = 255
    Image.fromarray(one, mode="L").save(good)

    assert len(read_instance_masks(case)) == 1  # control: one readable object, read

    planted = write(case / "masks" / "01.png", EXCLUDED)
    with Image.open(planted, formats=[EXCLUDED]) as handle:
        assert handle.size == (8, 8)  # control: a picture, just not a PNG

    with pytest.raises(InstanceError) as raised:
        read_instance_masks(case)
    assert "01.png" in str(raised.value)
    assert "named as a PNG" in str(raised.value)
