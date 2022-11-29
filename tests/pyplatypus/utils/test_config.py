from pyplatypus.utils.config import YamlConfigLoader, check_cv_tasks
from pathlib import Path
import yaml
import pytest

from pyplatypus.data_models.platypus_engine import PlatypusSolverInput
from pyplatypus.data_models.semantic_segmentation import (
    SemanticSegmentationData, SemanticSegmentationInput, SemanticSegmentationModelSpec
    )
from pyplatypus.data_models.object_detection import ObjectDetectionInput
from pyplatypus.data_models import augmentation as AM
from pyplatypus.data_models.optimizers import AdamSpec


class mocked_optimizer_spec:

    def __init__(*args, **kwargs):
        return None

    name = "optimizer_1"
    learning_rate = 0.1

class mocked_loss_spec:

    def __init__(*args, **kwargs):
        return None

    name = "loss_1"
    alpha = 0.5


class mocked_metric1_spec:

    def __init__(*args, **kwargs):
        return None

    name = "metric_1"
    alpha = 1

class mocked_metric2_spec:

    def __init__(*args, **kwargs):
        return None

    name = "metric_2"
    alpha = 2

class mocked_callback1_spec:

    def __init__(*args, **kwargs):
        return None

    name = "callback_1"


class mocked_callback2_spec:

    def __init__(*args, **kwargs):
        return None

    name = "callback_2"


class mocked_terminate_on_nan:
    def __init__(*args, **kwargs):
        return None

    name = "TerminateOnNaN" 

class TestCheckCVTasks:
    config_none = dict({"random_tast_name": None, "semantic_segmenatation": None, "object_detection": None})
    config_notnone = dict({"random_tast_name": "task1", "semantic_segmentation": "task2"})
    test_data = [
        (config_none, []),
        (config_notnone, ["semantic_segmentation"])
    ]
    @pytest.mark.parametrize("input_config, tasks_list", test_data)
    def test_check_cv_tasks(self, input_config, tasks_list):
        assert check_cv_tasks(input_config) == tasks_list


class TestYAMLConfigLoader:
    @staticmethod
    def mock_config():
        mocked_config = {
            "semantic_segmentation": {
                "data": {
                    "train_path": 'tests/testdata/nested_dirs/',
                    "validation_path": 'tests/testdata/nested_dirs/',
                    "test_path": 'tests/testdata/nested_dirs/',
                    "colormap": [[0, 0, 0], [255, 255, 255]],
                    "mode": 'nested_dirs',
                    "shuffle": False,
                    "subdirs": ["images", "masks"],
                    "column_sep": ';'
                },
                "models":
                    [{
                        "name": 'res_u_net',
                        "net_h": 300,
                        "net_w": 300,
                        "h_splits": 0,
                        "w_splits": 0,
                        "grayscale": False,
                        "blocks": 4,
                        "n_class": 2,
                        "filters": 16,
                        "dropout": 0.2,
                        "batch_normalization": True,
                        "kernel_initializer": 'he_normal',
                        "resunet": True,
                        "linknet": False,
                        "plus_plus": False,
                        "deep_supervision": False,
                        "use_separable_conv2d": True,
                        "use_spatial_droput2d": True,
                        "use_up_sampling2d": False,
                        "u_net_conv_block_width": 4,
                        "activation_layer": "relu",
                        "batch_size": 32,
                        "epochs": 100,
                        "loss": 'focal loss',
                        "metrics": ['tversky coefficient', 'iou coefficient'],
                        "optimizer": {"Adam": {}},
                        "augmentation": {
                            "InvertImg": {
                                "always_apply": True,
                                "p": 1
                            }
                        }
        
                    }]
                }
            }
        return mocked_config

    def mocked_solver_datamodel(*args, **kwargs):
        return {}

    def test_initialize_invalid_path(self):
        with pytest.raises(NotADirectoryError):
            YamlConfigLoader(Path("/noexistent_file"))

    def test_initialize_valid_path(self, tmpdir):
        mocked_file_path = Path(tmpdir)/"mocked_file.txt"
        with open(mocked_file_path, "w") as mocked_file:
            mocked_file.writelines("")
        ycl = YamlConfigLoader(mocked_file_path)
        assert ycl.config_yaml_path == mocked_file_path

    def test_load_config_from_yaml(self, tmpdir):
        mocked_file_path = Path(tmpdir)/"config_file.yaml"
        with open(mocked_file_path, "w") as mocked_file:
            yaml.dump(self.mock_config(), mocked_file)

        assert self.mock_config() == YamlConfigLoader.load_config_from_yaml(mocked_file_path)

    def test_create_semantic_segmentation_config(self):
        config = self.mock_config()
        parsed_config = YamlConfigLoader("").create_semantic_segmentation_config(config)
        assert parsed_config.data == SemanticSegmentationData(**config.get("semantic_segmentation").get("data"))
        assert parsed_config.models[0] == SemanticSegmentationModelSpec(**config.get("semantic_segmentation").get("models")[0])
        assert isinstance(parsed_config, SemanticSegmentationInput)

    def test_create_semantic_segmentation_config_no_optimizer(self):
        config = self.mock_config()
        config.get("semantic_segmentation").get("models")[0].pop("optimizer")
        parsed_config = YamlConfigLoader("").create_semantic_segmentation_config(config)
        assert parsed_config.models[0].optimizer == AdamSpec()

    def test_create_semantic_segmentation_config_empty_optimizer(self):
        config = self.mock_config()
        config.get("semantic_segmentation").get("models")[0].update({"optimizer": None})
        parsed_config = YamlConfigLoader("").create_semantic_segmentation_config(config)
        assert parsed_config.models[0].optimizer == AdamSpec()

    def test_create_object_detection_config(self):
        config = self.mock_config()
        assert YamlConfigLoader.create_object_detection_config(config) == ObjectDetectionInput()

    def test_process_optimizer_field(self, monkeypatch):
        monkeypatch.setattr("pyplatypus.utils.config.OM.OptimizerNameSpec", mocked_optimizer_spec, raising=False)
        config = self.mock_config()
        config.update(optimizer={"OptimizerName": {}})
        processed_config = YamlConfigLoader.process_optimizer_field(config)
        assert processed_config.get("optimizer").name == "optimizer_1"
        assert processed_config.get("optimizer").learning_rate == 0.1

    def test_process_optimizer_field_empty_field(self):
        config = self.mock_config()
        config.update(optimizer=None)
        processed_config = YamlConfigLoader.process_optimizer_field(config)
        assert "optimizer" not in processed_config.keys()

    def test_process_callbacks_field_list(self, monkeypatch):
        monkeypatch.setattr("pyplatypus.utils.config.CM.Callback1Spec", mocked_callback1_spec, raising=False)
        monkeypatch.setattr("pyplatypus.utils.config.CM.Callback2Spec", mocked_callback2_spec, raising=False)
        monkeypatch.setattr("pyplatypus.utils.config.available_callbacks_without_specification", ["Callback1", "Callback2"], raising=False)
        config = self.mock_config()
        config.update(callbacks=["Callback1", "Callback2"])
        processed_config = YamlConfigLoader.process_callbacks_field(config)
        assert set([callback.name for callback in processed_config.get("callbacks")]) == set(["callback_1", "callback_2"])

    def test_process_callbacks_field_dict(self, monkeypatch):
        monkeypatch.setattr("pyplatypus.utils.config.CM.Callback1Spec", mocked_callback1_spec, raising=False)
        monkeypatch.setattr("pyplatypus.utils.config.CM.Callback2Spec", mocked_callback2_spec, raising=False)
        monkeypatch.setattr("pyplatypus.utils.config.CM.TerminateOnNaNSpec", mocked_terminate_on_nan, raising=False)
        config = self.mock_config()
        config.update(callbacks={"Callback1": {}, "Callback2": {}, "TerminateOnNaN": None})
        processed_config = YamlConfigLoader.process_callbacks_field(config)
        assert set([callback.name for callback in processed_config.get("callbacks")]) == set(["callback_1", "callback_2", "TerminateOnNaN"])

    def test_process_callbacks_field_wrong_structure(self):
        config = self.mock_config()
        config.update(callbacks="callbacks")
        with pytest.raises(ValueError):
            processed_config = YamlConfigLoader.process_callbacks_field(config)

    def test_process_callbacks_field_dict(self, monkeypatch):
        config = self.mock_config()
        config.update(callbacks=None)
        processed_config = YamlConfigLoader.process_callbacks_field(config)
        assert "callbacks" not in processed_config.keys()

    def test_load(self, mocker):
        ycl_path = "pyplatypus.utils.config.YamlConfigLoader."
        mocker.patch(ycl_path + "load_config_from_yaml", return_value=None)
        mocker.patch(ycl_path + "create_semantic_segmentation_config", return_value=None)
        mocker.patch(ycl_path + "create_object_detection_config", return_value=None)
        mocker.patch("pyplatypus.utils.config.PlatypusSolverInput", self.mocked_solver_datamodel)
        assert YamlConfigLoader(Path("")).load() == {}


    def test_process_loss_field_dict(self, monkeypatch):
        monkeypatch.setattr("pyplatypus.utils.config.SSLM.LossNameSpec", mocked_loss_spec, raising=False)
        config = self.mock_config()
        config.update(loss={"loss_name": {}})
        processed_config = YamlConfigLoader.process_loss_field(config)
        assert processed_config.get("loss").name == "loss_1"
        assert processed_config.get("loss").alpha == 0.5

    def test_process_loss_field_str(self, monkeypatch):
        monkeypatch.setattr("pyplatypus.utils.config.SSLM.LossNameSpec", mocked_loss_spec, raising=False)
        config = self.mock_config()
        config.update(loss="loss_name")
        processed_config = YamlConfigLoader.process_loss_field(config)
        assert processed_config.get("loss").name == "loss_1"
        assert processed_config.get("loss").alpha == 0.5

    def test_process_loss_field_wrong_type(self, monkeypatch):
        config = self.mock_config()
        config.update(loss=["loss_name"])
        with pytest.raises(ValueError):
            processed_config = YamlConfigLoader.process_loss_field(config)

    def test_process_loss_field_empty_field(self):
        config = self.mock_config()
        config.update(loss=None)
        processed_config = YamlConfigLoader.process_loss_field(config)
        assert "loss" not in processed_config.keys()

    def test_process_loss_field_list(self, monkeypatch):
        monkeypatch.setattr("pyplatypus.utils.config.SSLM.Metric1Spec", mocked_metric1_spec, raising=False)
        monkeypatch.setattr("pyplatypus.utils.config.SSLM.Metric2Spec", mocked_metric2_spec, raising=False)
        config = self.mock_config()
        config.update(metrics=["metric_1", "metric_2"])
        processed_config = YamlConfigLoader.process_metrics_field(config)
        assert processed_config.get("metrics")[0].name in ["metric_1", "metric_2"]
        assert processed_config.get("metrics")[1].name in ["metric_1", "metric_2"]

    def test_process_metrics_field_dict(self, monkeypatch):
        monkeypatch.setattr("pyplatypus.utils.config.SSLM.Metric1Spec", mocked_metric1_spec, raising=False)
        monkeypatch.setattr("pyplatypus.utils.config.SSLM.Metric2Spec", mocked_metric2_spec, raising=False)
        config = self.mock_config()
        config.update(metrics={"metric_1": {}, "metric_2": {}})
        processed_config = YamlConfigLoader.process_metrics_field(config)
        assert processed_config.get("metrics")[0].name == "metric_1"
        assert processed_config.get("metrics")[1].name == "metric_2"

    def test_process_metrics_field_wrong_structure(self):
        config = self.mock_config()
        config.update(metrics="callbacks")
        with pytest.raises(ValueError):
            processed_config = YamlConfigLoader.process_metrics_field(config)

    def test_process_augmentation_field_dict(self, monkeypatch):
        config = self.mock_config()
        config.update(augmentation=None)
        processed_config = YamlConfigLoader.process_augmentation_field(config)
        assert "augmentation" not in processed_config.keys()

    def test_process_augmentation_field_dict(self, monkeypatch):
        monkeypatch.setattr("pyplatypus.utils.config.AM.Metric1Spec", mocked_metric1_spec, raising=False)
        monkeypatch.setattr("pyplatypus.utils.config.AM.Metric2Spec", mocked_metric2_spec, raising=False)
        config = self.mock_config()
        config.update(augmentation={"Metric1": {}, "Metric2": {}})
        processed_config = YamlConfigLoader.process_augmentation_field(config)
        assert processed_config.get("augmentation")[0].name == "metric_1"
        assert processed_config.get("augmentation")[1].name == "metric_2"

    def test_process_augmentation_field_wrong_structure(self):
        config = self.mock_config()
        config.update(augmentation="augmentation")
        with pytest.raises(ValueError):
            processed_config = YamlConfigLoader.process_augmentation_field(config)
