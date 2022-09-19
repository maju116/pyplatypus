from platypus.utils.config_processing_functions import YamlConfigLoader, check_cv_tasks
from pathlib import Path
import yaml
import pytest


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
                "column_sep": ';',
                "loss": 'focal loss',
                "metrics": ['tversky coefficient', 'iou coefficient'],
                "optimizer": 'adam'
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
                    "epochs": 100
                }]
            },
    "augmentation": {
        "InvertImg": {
            "always_apply": True,
            "p": 1
        }
    }
    }

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
            yaml.dump(self.mocked_config, mocked_file)

        assert self.mocked_config == YamlConfigLoader.load_config_from_yaml(mocked_file_path)
