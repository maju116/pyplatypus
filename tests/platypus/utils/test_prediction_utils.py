from platypus.utils.prediction_utils import save_masks
from pathlib import Path
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
        assert isinstance(mask_path, Path)

def mocked_image_fromarray(image):
    return MockedPILImage(image)


def test_save_masks_wrong_type(monkeypatch):
    monkeypatch.setattr("platypus.utils.prediction_utils.Path.mkdir", mocked_path_mkdir, raising=False)
    with pytest.raises(NotImplementedError):
        save_masks(image_masks=["mask1"], paths=["directory/path1.dcm"], model_name="model1")

def test_save_masks_wrong_type(monkeypatch):
    monkeypatch.setattr("platypus.utils.prediction_utils.Path.mkdir", mocked_path_mkdir, raising=False)
    monkeypatch.setattr("platypus.utils.prediction_utils.Image.fromarray", mocked_image_fromarray)
    save_masks(image_masks=[np.array([0, 1, 1, 0]).reshape((1, 2, 2))], paths=["directory/path1.png"], model_name="model1")
