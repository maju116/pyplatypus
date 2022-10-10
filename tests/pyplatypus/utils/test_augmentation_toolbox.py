from pyplatypus.utils.augmentation_toolbox import (
    filter_out_incorrect_methods, create_augmentation_pipeline, prepare_augmentation_pipelines
    )
import albumentations as A
import mock, pytest


class mocked_augmentation_spec:
    name = "method_name1"

    def copy(self):
        return self
    
    def dict(self):
        return {"name": "method_name1"}


class mocked_config:
    def __init__(self, empty: bool):
        self.augmentation = [mocked_augmentation_spec()] if not empty else None


class TestFilterOutIncorrectMethods:
    def test_filter_out_incorrect_methods_train_no_valid_methods(self, mocker):
        with mock.patch("pyplatypus.utils.augmentation_toolbox.train_available_methods", ["method_name5"]):
            assert filter_out_incorrect_methods([mocked_augmentation_spec()], train=True) == []

    
    def test_filter_out_incorrect_methods_train_some_valid_methods(self, mocker):
        with mock.patch("pyplatypus.utils.augmentation_toolbox.train_available_methods", ["method_name2", "method_name1"]):
            assert filter_out_incorrect_methods([mocked_augmentation_spec()], train=True)[0].name == mocked_augmentation_spec().name

        
    def test_filter_out_incorrect_methods_validation_no_valid_methods(self, mocker):
        with mock.patch("pyplatypus.utils.augmentation_toolbox.validation_test_available_methods", ["method_name2", "method_name1"]):
            assert filter_out_incorrect_methods([mocked_augmentation_spec()], train=False)[0].name == mocked_augmentation_spec.name

class TestCreateAugmentationPipeline:
    
    augmentation_dict = {
        "method_name1": {
                "field0": 0,
                "field1": 1
        }
    }

    @staticmethod
    def mocked_filter_out_incorrect_methods(augmentation_dict: dict, train: bool):
        return [mocked_augmentation_spec]

    @staticmethod
    def mocked_method1():
        return "method_name1"

    @staticmethod
    def mocked_compose(pipes: list, p: float):
        return ["method_name1"]

    def test_create_augmentation_pipeline(self, monkeypatch, mocker):
        mocker.patch(
            "pyplatypus.utils.augmentation_toolbox.filter_out_incorrect_methods",
            return_value=[mocked_augmentation_spec()]
            )
        albumentations_path = "pyplatypus.utils.augmentation_toolbox.A"
        monkeypatch.setattr(
            albumentations_path+".method_name1", self.mocked_method1, raising=False
            )
        monkeypatch.setattr(albumentations_path+".Compose", self.mocked_compose)
        assert create_augmentation_pipeline([mocked_augmentation_spec()], train=True) == ["method_name1"]


class TestPrepareAugmentationPipelines:
    def test_prepare_augmentation_pipelines_no_augmentation(self):
        assert prepare_augmentation_pipelines(mocked_config(empty=True)) == (None, None)

    def test_prepare_augmentation_pipelines(self, mocker):
        mocker.patch("pyplatypus.utils.augmentation_toolbox.create_augmentation_pipeline", return_value="pipeline")
        assert prepare_augmentation_pipelines(mocked_config(empty=False)) == ("pipeline", "pipeline")
