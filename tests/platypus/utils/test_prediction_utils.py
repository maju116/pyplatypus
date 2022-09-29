from pyplatypus.utils.prediction_utils import save_masks
from pathlib import Path
from distutils.dir_util import copy_tree
import numpy as np
import pytest


def mocked_path_mkdir(masks_path: Path, exist_ok: bool):
    return None

class MockedPILImage:
    def __init__(self, image: np.array):
        self.image = image
        assert isinstance(image, np.ndarray)

    def convert(self, fmt: str):
        assert isinstance(fmt, str)
        return self

    def save(self, mask_path: Path):
        Path.mkdir(mask_path)

def mocked_image_fromarray(image):
    return MockedPILImage(image)


def test_save_masks_wrong_type(monkeypatch):
    monkeypatch.setattr("pyplatypus.utils.prediction_utils.Path.mkdir", mocked_path_mkdir, raising=False)
    with pytest.raises(NotImplementedError):
        save_masks(image_masks=["mask1"], paths=["directory/path1.dcm"], model_name="model1")

@pytest.mark.parametrize("custom_data_path", [(None, "custom_path")])
def test_save_masks_nested(tmpdir, monkeypatch, custom_data_path):
    tmp_dir = Path(tmpdir/"custom_path/")
    copy_tree("tests/testdata/nested_dirs/", str(tmp_dir))
    paths = [tmp_dir/"image_1/images/test_image_1.png", tmp_dir/"image_2/images/test_image_2.png", tmp_dir/"image_3/images/test_image_3.png"]
    
    monkeypatch.setattr("pyplatypus.utils.prediction_utils.Image.fromarray", mocked_image_fromarray)
    save_masks(image_masks=[np.array([0, 1, 1, 0]).reshape((1, 2, 2))]*3, paths=paths, model_name="model1", mode="nested_dirs")
    assert all([(Path(path).parents[1]/f"predicted_masks/{path.name.split('.')[0]}_model1_predicted_mask.png").exists() for path in paths])

@pytest.mark.parametrize("custom_data_path", [(None, "custom_path")])
def test_save_masks_not_nested(tmpdir, monkeypatch, custom_data_path):
    tmp_dir = Path(tmpdir/"custom_path/")
    copy_tree("tests/testdata/dir", str(tmp_dir))
    paths = [tmp_dir/"images/test_image_1.png", tmp_dir/"images/test_image_2.png", tmp_dir/"images/test_image_3.png"]
    
    monkeypatch.setattr("pyplatypus.utils.prediction_utils.Image.fromarray", mocked_image_fromarray)
    save_masks(image_masks=[np.array([0, 1, 1, 0]).reshape((1, 2, 2))]*3, paths=paths, model_name="model1", mode="config_file")
    assert all([(Path(path).parents[0]/f"predicted_masks/{path.name.split('.')[0]}_model1_predicted_mask.png").exists() for path in paths])