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

    def evaluate(x):
        return [.1, .2, .3]

    def fit(self, *args, **kwargs):
        return pd.DataFrame({"history": [0, 1, 2]})


class mocked_loss_spec:
    name = "loss_name"


class mocked_metric_spec:
    name = "metric_name"


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
    mocked_training_history = pd.DataFrame({"history": [0, 1, 2]})
    config = PlatypusSolverInput(
        object_detection=ObjectDetectionInput(),
        semantic_segmentation=SemanticSegmentationInput(
            data=SemanticSegmentationData(**{
                "train_path": "tests/",
                "validation_path": "tests/",
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
    initialized_engine = PlatypusEngine(config=config.copy())
    engine_path = "pyplatypus.engine.PlatypusEngine"

    @staticmethod
    def mocked_save_masks(image_masks, paths, model_name, mode):
        assert isinstance(image_masks, list)
        assert isinstance(paths, list)
        assert model_name == "model1"

    def mocked_produce_and_save_predicted_masks_for_model(self, model_name, custom_data_path, task_type):
        assert isinstance(model_name, str)

    def mocked_update_cache(self, model_name, model, training_history, model_specification):
        self.results = (model_name, model, self.mocked_training_history, model_specification)

    def mock_build_and_train_segmentation_models(self, train_augmentation_pipeline, validation_augmentation_pipeline):
        self.model = "trained_model"

    def test_init(self):
        initialized_engine = PlatypusEngine(config=self.config.copy())
        assert initialized_engine.config == dict(self.config)
        assert initialized_engine.cache == {}

    def test_update_cache(self):
        engine = self.initialized_engine
        engine.cache.update(semantic_segmentation={})
        engine.update_cache(
            model_name="model_name", model="model", training_history="training_history",
            model_specification={"model_type": "u_shaped"}
        )
        assert engine.cache == {"semantic_segmentation":
            {"model_name": {
                "model": "model",
                "training_history": "training_history",
                "model_specification": {"model_type": "u_shaped"}
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
        mocker.patch("pyplatypus.engine.prepare_data_generator", return_value=(
            mocked_generator
        ))
        mocker.patch(self.engine_path + ".compile_u_shaped_model", return_value=mocked_u_shaped_model)
        mocker.patch(self.engine_path + ".update_cache", self.mocked_update_cache)
        engine.build_and_train_segmentation_models(None, None)
        assert self.results == (
            "model_name", mocked_u_shaped_model, self.mocked_training_history,
            self.config.semantic_segmentation.models[0].dict()
        )

    def test_compile_u_shaped_model(self, mocker):
        model_cfg = self.config.semantic_segmentation.models[0]
        mocker.patch("pyplatypus.engine.u_shaped_model", return_value=mocked_u_shaped_model)
        mocker.patch("pyplatypus.engine.prepare_loss_and_metrics", return_value=(None, None))
        mocker.patch("pyplatypus.engine.prepare_optimizer", return_value=None)
        self.initialized_engine.compile_u_shaped_model(model_cfg=model_cfg)
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
        mocker.patch(self.engine_path + ".predict_based_on_generator",
                     return_value=(["predictions"], ["paths"], "colormap", "mode"))
        mocker.patch("pyplatypus.engine.transform_probabilities_into_binaries", return_value="prediction_binary")
        mocker.patch("pyplatypus.engine.concatenate_binary_masks", return_value="prediction_mask")
        mocker.patch("pyplatypus.engine.save_masks", self.mocked_save_masks)
        self.initialized_engine.produce_and_save_predicted_masks(model_name="model1")

    @pytest.mark.parametrize("custom_data_path", [(None), ("some_path")])
    def test_predict_based_on_generator(self, mocker, custom_data_path):
        mocker.patch("pyplatypus.engine.predict_from_generator", return_value=("predictions", "paths"))
        mocker.patch("pyplatypus.engine.prepare_data_generator", return_value=mocked_generator)
        engine = self.initialized_engine
        engine.cache = {"semantic_segmentation": {"model_name": {"model": "model", "data_generator": mocked_generator}}}
        assert engine.predict_based_on_generator(
            model_name="model_name", custom_data_path=custom_data_path
        ) == ("predictions", "paths", mocked_generator.colormap, mocked_generator.mode)


    def test_prepare_evaluation_results(self):
        evaluation_metrics = [.1, .2, .3]
        evaluation_columns = ["model_name", "loss", "metric_1", "metric_2"]
        prepared_evaluation_metrics = PlatypusEngine.prepare_evaluation_results(evaluation_metrics, "model_name", evaluation_columns)
        pd.testing.assert_frame_equal(prepared_evaluation_metrics, pd.DataFrame({
            "model_name": "model_name", "loss": .1, "metric_1": .2, "metric_2": .3
        }, index=[0]))


    @pytest.mark.parametrize("custom_data_path", [(None), ("some_path")])
    def test_evaluate_based_on_generator(self, mocker, custom_data_path):
        mocker.patch("pyplatypus.engine.prepare_data_generator", return_value="generator")
        engine = self.initialized_engine
        engine.cache = {"semantic_segmentation": {"model_name": {"model": mocked_u_shaped_model, "validation_generator": mocked_generator}}}
        assert engine.evaluate_based_on_generator(
            model_name="model_name", model_cfg="model_cfg", custom_data_path=custom_data_path
        ) == [.1, .2, .3]

    def test_prepare_evaluation_table(self, monkeypatch):
        monkeypatch.setattr("pyplatypus.data_models.semantic_segmentation_datamodel.implemented_losses", ["loss_name"], raising=False)
        monkeypatch.setattr("pyplatypus.data_models.semantic_segmentation_datamodel.implemented_metrics", ["metric_name"], raising=False)
        model_cfg = SemanticSegmentationModelSpec(
            name="model_name", net_h=300, net_w=300, blocks=2, n_class=2, filters=5, dropout=.1,
            loss=mocked_loss_spec, metrics=[mocked_metric_spec]
            )
        assert all(PlatypusEngine.prepare_evaluation_table(model_cfg).columns == ["model_name", "loss_name", "categorical_crossentropy", "metric_name"])

    
    def test_evaluate_model(self, mocker):
        task_type = "semantic_segmentation"
        model_name = "model_name"
        cache = {
            task_type: 
                {model_name: {"model_specification": {}}}
            }
        engine = self.initialized_engine
        engine.cache = cache
        mocker.patch(self.engine_path + ".prepare_evaluation_table", return_value=pd.DataFrame(columns=["col1", "col2"]))
        mocker.patch(self.engine_path + ".evaluate_based_on_generator", return_value=None)
        mocker.patch(self.engine_path + ".prepare_evaluation_results", return_value=[.1, .2, .3])
        assert engine.evaluate_model(model_name, task_type) == [.1, .2, .3]


    @pytest.mark.parametrize("model_name, result", [("model1", ["evaluation_results"]), (None, ["evaluation_results"]*2)])
    def test_evaluate_models(self, mocker, model_name, result):
        mocker.patch(self.engine_path + ".get_model_names", return_value=["model_name1", "model_name2"])
        mocker.patch(self.engine_path + ".evaluate_model", return_value = "evaluation_results")
        assert self.initialized_engine.evaluate_models(model_name) == result
