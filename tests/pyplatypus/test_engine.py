from pyplatypus.engine import PlatypusEngine
from pyplatypus.data_models.platypus_engine_datamodel import PlatypusSolverInput
from pyplatypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationData, SemanticSegmentationInput, \
    SemanticSegmentationModelSpec
from pyplatypus.data_models.object_detection_datamodel import ObjectDetectionInput
from pyplatypus.data_models.augmentation_datamodel import AugmentationSpecFull
from pyplatypus.data_models.optimizer_datamodel import AdamSpec
import pytest
import pandas as pd


class mocked_model:
    def compile(**kwargs):
        return None


class mocked_u_shaped_model:
    model = mocked_model

    def fit(self, *args, **kwargs):
        return pd.DataFrame({"history": [0, 1, 2]})


mocked_training_history = pd.DataFrame({"history": [0, 1, 2]})


class mocked_generator:
    steps_per_epoch = 10
    colormap = [(0, 0, 0), (255, 255, 255)]
    mode = "nested_dirs"
    only_images = False
    subdirs = ["masks", "images"]
    column_sep = ";"
    batch_size = 1

    def create_images_masks_paths(self, *args):
        return {"images_paths": ["path1", "path2"]}


class TestPlatypusEngine:
    config = PlatypusSolverInput(
        object_detection=ObjectDetectionInput(),
        semantic_segmentation=SemanticSegmentationInput(
            data=SemanticSegmentationData(**{
                "train_path": "tests/",
                "validation_path": "tests/",
                "test_path": "tests/",
                "colormap": [[0, 0, 0]],
                "mode": "nested_dirs",
                "shuffle": True,
                "subdirs": ["images", "masks"],
                "column_sep": ","
            }),
            models=[SemanticSegmentationModelSpec(**{
                "name": "model_name",
                "net_h": 8,
                "net_w": 8,
                "blocks": 4,
                "n_class": 2,
                "filters": 2,
                "dropout": .1,
                "optimizer": AdamSpec()
            })]
        ),
        augmentation=AugmentationSpecFull()
    )
    initialized_engine = PlatypusEngine(config=config.copy(), cache={})
    engine_path = "pyplatypus.engine.PlatypusEngine"

    @staticmethod
    def mocked_save_masks(image_masks, paths, model_name, mode):
        assert isinstance(image_masks, list)
        assert isinstance(paths, list)
        assert model_name == "model1"

    def mocked_produce_and_save_predicted_masks_for_model(self, model_name, task_type):
        assert isinstance(model_name, str)

    def mocked_update_cache(self, model_name, model, training_history, model_specification, generator):
        self.results = (model_name, model, mocked_training_history, model_specification, generator)

    def mock_build_and_train_segmentation_models(self, train_augmentation_pipeline, validation_augmentation_pipeline):
        self.model = "trained_model"

    def test_init(self):
        initialized_engine = PlatypusEngine(config=self.config.copy(), cache={})
        assert initialized_engine.config == dict(self.config)
        assert initialized_engine.cache == {}

    def test_update_cache(self):
        engine = self.initialized_engine
        engine.cache.update(semantic_segmentation={})
        engine.update_cache(
            model_name="model_name", model="model", training_history="training_history",
            model_specification={"model_type": "u_shaped"}, generator="generator"
        )
        assert engine.cache == {"semantic_segmentation":
            {"model_name": {
                "model": "model",
                "training_history": "training_history",
                "model_specification": {"model_type": "u_shaped"},
                "data_generator": "generator"
            }}
        }

    def test_train(self, mocker, monkeypatch):
        engine = self.initialized_engine
        mocker.patch("pyplatypus.engine.prepare_augmentation_pipelines", return_value=(None, None))
        mocker.patch(self.engine_path + ".build_and_train_segmentation_models",
                     self.mock_build_and_train_segmentation_models)
        engine.train()
        assert self.model == "trained_model"

    def test_build_and_train_segmentation_model(self, mocker):
        engine = self.initialized_engine
        mocker.patch("pyplatypus.engine.prepare_data_generators", return_value=(
            mocked_generator, mocked_generator, mocked_generator
        ))
        mocker.patch(self.engine_path + ".compile_u_shaped_model", return_value=mocked_u_shaped_model)
        mocker.patch(self.engine_path + ".update_cache", self.mocked_update_cache)
        engine.build_and_train_segmentation_models(None, None)
        assert self.results == (
            "model_name", mocked_u_shaped_model, mocked_training_history,
            self.config.semantic_segmentation.models[0].dict(), mocked_generator
        )

    def test_compile_u_shaped_model(self, mocker):
        model_cfg = self.config.semantic_segmentation.models[0]
        mocker.patch("pyplatypus.engine.u_shaped_model", return_value=mocked_u_shaped_model)
        mocker.patch("pyplatypus.engine.prepare_loss_and_metrics", return_value=(None, None))
        mocker.patch("pyplatypus.engine.prepare_optimizer", return_value=None)
        self.initialized_engine.compile_u_shaped_model(model_cfg=model_cfg,
                                                       segmentation_spec=self.config.semantic_segmentation)
        assert True

    def test_get_model_names(self):
        model_names = self.initialized_engine.get_model_names(config=dict(self.config),
                                                              task_type="semantic_segmentation")
        assert model_names == ["model_name"]

    @pytest.mark.parametrize("model_name", [("model1"), (None)])
    def test_produce_and_save_predicted_masks(self, mocker, model_name):
        mocker.patch(self.engine_path + ".get_model_names", return_value=["model_name1", "model_name2"])
        mocker.patch(self.engine_path + ".produce_and_save_predicted_masks_for_model",
                     self.mocked_produce_and_save_predicted_masks_for_model)
        self.initialized_engine.produce_and_save_predicted_masks(model_name)

    def test_produce_and_save_predicted_masks_for_model(self, mocker):
        mocker.patch(self.engine_path + ".predict_based_on_test_generator",
                     return_value=(["predictions"], ["paths"], "colormap", "mode"))
        mocker.patch("pyplatypus.engine.transform_probabilities_into_binaries", return_value="prediction_binary")
        mocker.patch("pyplatypus.engine.concatenate_binary_masks", return_value="prediction_mask")
        mocker.patch("pyplatypus.engine.save_masks", self.mocked_save_masks)
        self.initialized_engine.produce_and_save_predicted_masks(model_name="model1")

    @pytest.mark.parametrize("custom_data_path", [(None), ("some_path")])
    def test_predict_based_on_test_generator(self, mocker, custom_data_path):
        mocker.patch("pyplatypus.engine.predict_from_generator", return_value=("predictions", "paths"))
        engine = self.initialized_engine
        engine.cache = {"semantic_segmentation": {"model_name": {"model": "model", "data_generator": mocked_generator}}}
        assert engine.predict_based_on_test_generator(
            model_name="model_name", custom_data_path=custom_data_path
        ) == ("predictions", "paths", mocked_generator.colormap, mocked_generator.mode)
