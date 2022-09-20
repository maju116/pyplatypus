from platypus.utils.augmentation import filter_out_incorrect_methods, create_augmentation_pipeline
import albumentations as A
import mock, pytest


class TestFilterOutIncorrectMethods:
    augmentation_dict = {
        "method_name1": "method_spec1",
        "method_name2": "method_spec2",
        "method_name3": "method_spec3",
        "method_name4": "method_spec4"
    }
    def test_filter_out_incorrect_methods_train_no_valid_methods(self, mocker):
        with mock.patch("platypus.utils.augmentation.train_available_methods", ["method_name5"]):
            assert filter_out_incorrect_methods(self.augmentation_dict, train=True) == []

    
    def test_filter_out_incorrect_methods_train_some_valid_methods(self, mocker):
        with mock.patch("platypus.utils.augmentation.train_available_methods", ["method_name2", "method_name1"]):
            assert filter_out_incorrect_methods(self.augmentation_dict, train=True) == ["method_name1", "method_name2"]

        
    def test_filter_out_incorrect_methods_validation_no_valid_methods(self, mocker):
        with mock.patch("platypus.utils.augmentation.validation_test_available_methods", ["method_name2", "method_name1"]):
            assert filter_out_incorrect_methods(self.augmentation_dict, train=False) == ["method_name1", "method_name2"]

class TestCreateAugmentationPipeline:
    
    augmentation_dict = {
        "method_name1": {
                "field0": 0,
                "field1": 1
        }
    }

    @staticmethod
    def mocked_filter_out_incorrect_methods(augmentation_dict: dict, train: bool):
        return ["method_name1"]

    @staticmethod
    def mocked_method1(field0: int, field1: int):
        return "method_name1"

    @staticmethod
    def mocked_compose(pipes: list, p: float):
        return ["method_name1"]

    def test_create_augmentation_pipeline(self, monkeypatch):
        monkeypatch.setattr(
            "platypus.utils.augmentation.filter_out_incorrect_methods",
            self.mocked_filter_out_incorrect_methods
            )
        albumentations_path = "platypus.utils.augmentation.A"
        monkeypatch.setattr(
            albumentations_path+".method_name1", self.mocked_method1, raising=False
            )
        monkeypatch.setattr(albumentations_path+".Compose", self.mocked_compose)
        assert create_augmentation_pipeline(self.augmentation_dict, train=True) == ["method_name1"]
