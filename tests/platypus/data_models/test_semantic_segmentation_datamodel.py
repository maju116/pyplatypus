from pyplatypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationData, SemanticSegmentationModelSpec
from pathlib import Path
import pytest


class TestSemanticSegmentationData:

    data_models_path = "pyplatypus.data_models.semantic_segmentation_datamodel"

    @staticmethod
    def create_input(tmpdir):
        ssd_data = {
            "train_path": f"{tmpdir}/synthetic_train_path",
            "validation_path": f"{tmpdir}/synthetic_validation_path",
            "test_path": f"{tmpdir}/synthetic_test_path",
            "colormap": [[0,0,0]],
            "mode": "mode1",
            "shuffle": True,
            "subdirs": ["images", "masks"],
            "column_sep": ",",
            "loss": "loss1",
            "metrics": ["metric1"],
            "optimizer": "optimizer1"
            }
        return ssd_data

    def mock_config(self, mocker):
        mocker.patch(self.data_models_path+".implemented_modes", ["mode1"])
        mocker.patch(self.data_models_path+".implemented_losses", ["loss1"])
        mocker.patch(self.data_models_path+".implemented_metrics", ["metric1"])
        mocker.patch(self.data_models_path+".implemented_optimizers", ["optimizer1"])
    
    @staticmethod
    def create_paths(*args, tmpdir):
        for path in args:
            Path.mkdir(Path(tmpdir)/path, exist_ok=True)

    @pytest.mark.parametrize("paths_to_create", [
        ("synthetic_validation_path", "synthetic_test_path"),
        ("synthetic_train_path", "synthetic_test_path"),
        ("synthetic_train_path", "synthetic_validation_path")
        ])
    def test_check_if_path_exists(self, mocker, tmpdir, paths_to_create):
        self.mock_config(mocker)
        self.create_paths(*paths_to_create, tmpdir=tmpdir)
        with pytest.raises(NotADirectoryError):
            ssd = SemanticSegmentationData(
                **self.create_input(tmpdir)
            )

    @pytest.mark.parametrize("data_update_expression", [
        ({"mode": "invalid_mode"}),
        ({"loss": "invalid_loss"}),
        ({"metrics": ["invalid_metrics"]}),
        ({"optimizer": "invalid_optimizer"})
        ])
    def test_check_mode_loss_metrics_optimizer_validators(self, mocker, tmpdir, data_update_expression):
        self.mock_config(mocker)
        self.create_paths("synthetic_train_path", "synthetic_validation_path", "synthetic_test_path", tmpdir=tmpdir)
        data = self.create_input(tmpdir)
        data.update(data_update_expression)
        with pytest.raises(ValueError):
            ssd = SemanticSegmentationData(
                **data
            )

    def test_colormap_validator(self, mocker, tmpdir):
        self.mock_config(mocker)
        self.create_paths("synthetic_train_path", "synthetic_validation_path", "synthetic_test_path", tmpdir=tmpdir)
        data = self.create_input(tmpdir)
        data.update(colormap=[(0), (1)])
        with pytest.raises(ValueError):
            ssd = SemanticSegmentationData(
                **data
            )

class TestSemanticSegmentationModelSpec:

    def test_activation_layer_validator(self, mocker):
        model_spec = {
            "name": "model_name",
            "net_h": 8,
            "net_w": 8,
            "blocks": 2,
            "n_class": 2,
            "filters": 2,
            "dropout": 0.1,
            "activation_layer": "invalid_activation"
        }
        mocker.patch("pyplatypus.data_models.semantic_segmentation_datamodel.available_activations", ["valid_activation"])
        with pytest.raises(ValueError):
            SemanticSegmentationModelSpec(**model_spec)
