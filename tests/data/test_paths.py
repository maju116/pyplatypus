import pytest

from pyplatypus import ConfigError
from pyplatypus.data import discover
from pyplatypus.spec.common import DataMode


def test_nested_dirs_finds_every_sample(nested_root, binary_data):
    found = discover(nested_root, binary_data)
    assert len(found) == 3
    assert [s.key for s in found.samples] == ["sample_0", "sample_1", "sample_2"]
    assert len(found.samples[0].masks) == 2


def test_samples_are_sorted(nested_root, binary_data):
    """Order has to be stable or a seed does not reproduce a run."""
    first = [s.key for s in discover(nested_root, binary_data).samples]
    second = [s.key for s in discover(nested_root, binary_data).samples]
    assert first == second == sorted(first)


def test_only_images_skips_masks(nested_root, binary_data):
    found = discover(nested_root, binary_data, only_images=True)
    assert all(s.masks == () for s in found.samples)


def test_incomplete_sample_is_an_error_by_default(nested_root, binary_data):
    """The old package logged a warning and dropped the sample. Warnings in a loop over
    536 directories are warnings nobody reads."""
    for mask in (nested_root / "sample_1" / "masks").iterdir():
        mask.unlink()
    with pytest.raises(ConfigError) as caught:
        discover(nested_root, binary_data)
    assert "sample_1" in str(caught.value)
    assert "strict=False" in str(caught.value)


def test_incomplete_sample_can_be_skipped_on_request(nested_root, binary_data):
    for mask in (nested_root / "sample_1" / "masks").iterdir():
        mask.unlink()
    found = discover(nested_root, binary_data, strict=False)
    assert len(found) == 2
    assert found.skipped and found.skipped[0][0] == "sample_1"


def test_missing_root_says_so(tmp_path, binary_data):
    with pytest.raises(ConfigError, match="does not exist"):
        discover(tmp_path / "nope", binary_data)


def test_config_file_mode(tmp_path, nested_root, binary_data):
    csv = tmp_path / "config.csv"
    rows = ["images,masks"]
    for n in range(3):
        sample = nested_root / f"sample_{n}"
        masks = ";".join(str(p) for p in sorted((sample / "masks").iterdir()))
        rows.append(f"{sample / 'images' / f'{n}.png'},{masks}")
    csv.write_text("\n".join(rows))

    data = binary_data.model_copy(update={"mode": DataMode.CONFIG_FILE})
    found = discover(csv, data)
    assert len(found) == 3
    assert len(found.samples[0].masks) == 2


def test_config_file_paths_are_relative_to_the_csv(tmp_path, binary_data):
    """So a config file travels with its data instead of only working from one directory."""
    import numpy as np

    from tests.conftest import write_png

    write_png(tmp_path / "img" / "a.png", np.zeros((8, 8, 3)))
    write_png(tmp_path / "msk" / "a.png", np.zeros((8, 8, 3)))
    csv = tmp_path / "config.csv"
    csv.write_text("images,masks\nimg/a.png,msk/a.png\n")

    data = binary_data.model_copy(update={"mode": DataMode.CONFIG_FILE})
    sample = discover(csv, data).samples[0]
    assert sample.image.is_absolute() or sample.image.exists()
    assert sample.image.exists() and sample.masks[0].exists()


def test_config_file_needs_the_right_columns(tmp_path, binary_data):
    csv = tmp_path / "config.csv"
    csv.write_text("pictures,masks\na.png,b.png\n")
    data = binary_data.model_copy(update={"mode": DataMode.CONFIG_FILE})
    with pytest.raises(ConfigError, match="'images' column"):
        discover(csv, data)
