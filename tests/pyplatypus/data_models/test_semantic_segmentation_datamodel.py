from pyplatypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationData, SemanticSegmentationModelSpec, SemanticSegmentationInput
from pathlib import Path
import pytest


class mocked_optimizer_spec:
    def __init__(self, name):
        self.name = name

class TestSemanticSegmentationData:

    data_models_path = "pyplatypus.data_models.semantic_segmentation_datamodel"

    @staticmethod
    def create_input(tmpdir):
        ssd_data = {
            "train_path": f"{tmpdir}/synthetic_train_path",
            "validation_path": f"{tmpdir}/synthetic_validation_path",
            "colormap": [[0,0,0]],
            "mode": "mode1",
            "shuffle": True,
            "subdirs": ["images", "masks"],
            "column_sep": ","
            }
        return ssd_data

    def mock_config(self, mocker):
        mocker.patch(self.data_models_path+".implemented_modes", ["mode1"])

    @staticmethod
    def create_paths(*args, tmpdir):
        for path in args:
            Path.mkdir(Path(tmpdir)/path, exist_ok=True)

    @pytest.mark.parametrize("paths_to_create", [
        ("synthetic_validation_path", "synthetic_train_path"),
        ("synthetic_train_path", "synthetic_validation_path")
        ])
    def test_check_if_path_exists(self, mocker, tmpdir, paths_to_create):
        self.mock_config(mocker)
        self.create_paths(*paths_to_create, tmpdir=tmpdir)
        with pytest.raises(NotADirectoryError):
            ssd = SemanticSegmentationData(
                **self.create_input(tmpdir)
            )

    def test_check_mode(self, mocker, tmpdir):
        self.mock_config(mocker)
        self.create_paths("synthetic_train_path", "synthetic_validation_path", tmpdir=tmpdir)
        data = self.create_input(tmpdir)
        data.update({"mode": "invalid_mode"})
        with pytest.raises(ValueError):
            ssd = SemanticSegmentationData(
                **data
            )

    def test_colormap_validator(self, mocker, tmpdir):
        self.mock_config(mocker)
        self.create_paths("synthetic_train_path", "synthetic_validation_path", tmpdir=tmpdir)
        data = self.create_input(tmpdir)
        data.update(colormap=[(0), (1)])
        with pytest.raises(ValueError):
            ssd = SemanticSegmentationData(
                **data
            )

class TestSemanticSegmentationModelSpec:
    model_spec = {
        "name": "model_name",
        "net_h": 8,
        "net_w": 8,
        "blocks": 2,
        "n_class": 2,
        "filters": 2,
        "dropout": 0.1,
        "activation_layer": "invalid_activation",
        "loss": mocked_optimizer_spec(name="loss1"),
        "metrics": [mocked_optimizer_spec(name="metric1")],
        "optimizer": mocked_optimizer_spec(name="optimizer1")
    }

    def mock_config(self, mocker):
        mocker.patch(self.data_models_path+".implemented_modes", ["mode1"])
        mocker.patch(self.data_models_path+".implemented_losses", ["loss1"])
        mocker.patch(self.data_models_path+".implemented_metrics", ["metric1"])
        mocker.patch(self.data_models_path+".available_optimizers", ["optimizer1"])

    @pytest.mark.parametrize("data_update_expression", [
        ({"loss": mocked_optimizer_spec(name="invalid_loss")}),
        ({"metrics": mocked_optimizer_spec(name="invalid_metrics")}),
        ({"optimizer": mocked_optimizer_spec(name="invalid_name")})
        ])
    def test_check_mode_loss_metrics_optimizer_validators(self, mocker, tmpdir, data_update_expression):
        data = self.model_spec.copy()
        data.update(data_update_expression)
        mocker.patch("pyplatypus.data_models.semantic_segmentation_datamodel.available_activations", ["valid_activation"])
        with pytest.raises(ValueError):
            SemanticSegmentationModelSpec(**data)

    def test_activation_layer_validator(self, mocker):
        mocker.patch("pyplatypus.data_models.semantic_segmentation_datamodel.available_activations", ["valid_activation"])
        with pytest.raises(ValueError):
            SemanticSegmentationModelSpec(**self.model_spec)


class TestSemanticSegmentationInput:
    models = [{
        "name": "model_name",
        "net_h": 8,
        "net_w": 8,
        "blocks": 2,
        "n_class": 2,
        "filters": 2,
        "dropout": 0.1,
        "loss": mocked_optimizer_spec(name="loss1"),
        "metrics": [mocked_optimizer_spec(name="metric1")],
        "optimizer": mocked_optimizer_spec(name="optimizer1")
    },
    {
        "name": "model_name",
        "net_h": 8,
        "net_w": 8,
        "blocks": 2,
        "n_class": 2,
        "filters": 2,
        "dropout": 0.1,
        "loss": mocked_optimizer_spec(name="loss1"),
        "metrics": [mocked_optimizer_spec(name="metric1")],
        "optimizer": mocked_optimizer_spec(name="optimizer1")
    },
    ]
    data_models_path = "pyplatypus.data_models.semantic_segmentation_datamodel"

    ssd_data = {
        "train_path": "",
        "validation_path": "",
        "test_path": "",
        "colormap": [[0,0,0]],
        "mode": "mode1",
        "shuffle": True,
        "subdirs": ["images", "masks"],
        "column_sep": ","
        }

    def mock_config(self, mocker):
        mocker.patch(self.data_models_path+".implemented_modes", ["mode1"])
        mocker.patch(self.data_models_path+".implemented_losses", ["loss1"])
        mocker.patch(self.data_models_path+".implemented_metrics", ["metric1"])
        mocker.patch(self.data_models_path+".available_optimizers", ["optimizer1"])

    def test_validator(self, mocker, monkeypatch):
        self.mock_config(mocker)
        with pytest.raises(ValueError):
            input_segmentation = SemanticSegmentationInput(
                data=self.ssd_data,
                models=self.models
            )
