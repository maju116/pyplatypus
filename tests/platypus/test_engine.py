from platypus.engine import platypus_engine
from platypus.data_models.platypus_engine_datamodel import PlatypusSolverInput
from platypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationData, SemanticSegmentationInput, SemanticSegmentationModelSpec
from platypus.data_models.object_detection_datamodel import ObjectDetectionInput
from platypus.data_models.augmentation_datamodel import AugmentationSpecFull

class mocked_u_shaped_model:

    def fit(self, *args, **kwargs):
        return None
    
    def compile(self, *args, **kwargs):
        return None
        
class mocked_generator:
    steps_per_epoch = 10

class TestPlatypusEngine:
    config = PlatypusSolverInput(
        object_detection=ObjectDetectionInput(),
        semantic_segmentation=SemanticSegmentationInput(
            data=SemanticSegmentationData(**{
                "train_path": "tests/",
                "validation_path": "tests/",
                "test_path": "tests/",
                "colormap": [[0,0,0]],
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
                "dropout": .1
            })]
        ),
        augmentation=AugmentationSpecFull()
    )
    initialized_engine = platypus_engine(config=config.copy(), cache={})
    engine_path = "platypus.engine.platypus_engine"

    def mocked_update_cache(self, model_name, model, model_specification, generator):
        self.results = (model_name, model, model_specification, generator)

    def mock_build_and_train_segmentation_models(self, train_augmentation_pipeline, validation_augmentation_pipeline):
        self.model = "trained_model"

    def test_init(self):
        initialized_engine = platypus_engine(config=self.config.copy(), cache={})
        assert initialized_engine.config == dict(self.config)
        assert initialized_engine.cache == {}

    def test_update_cache(self):
        engine = self.initialized_engine
        engine.cache.update(semantic_segmentation={})
        engine.update_cache(
            model_name="model_name", model="model", model_specification={"model_type": "u_shaped"}, generator="generator"
        )
        assert engine.cache == {"semantic_segmentation":
            {"model_name": {
                "model": "model",
                "model_specification": {"model_type": "u_shaped"},
                "data_generator": "generator"
            }}
        }

    def test_train(self, mocker, monkeypatch):
        engine = self.initialized_engine
        mocker.patch("platypus.engine.prepare_augmentation_pipelines", return_value=(None, None))
        mocker.patch(self.engine_path + ".build_and_train_segmentation_models", self.mock_build_and_train_segmentation_models)
        engine.train()
        assert self.model == "trained_model"

    def test_build_and_train_segmentation_model(self, mocker):
        engine = self.initialized_engine
        mocker.patch("platypus.engine.prepare_data_generators", return_value=(
            mocked_generator, mocked_generator, mocked_generator
            ))
        mocker.patch(self.engine_path + ".compile_u_shaped_model", return_value=mocked_u_shaped_model)
        mocker.patch(self.engine_path + ".update_cache", self.mocked_update_cache)
        engine.build_and_train_segmentation_models(None, None)
        assert self.results == (
            "model_name", mocked_u_shaped_model, self.config.semantic_segmentation.models[0].dict(), mocked_generator
            )

    def test_compile_u_shaped_model(self, mocker):
        model_cfg = self.config.semantic_segmentation.models[0]
        mocker.patch(self.engine_path + ".compile_u_shaped_model", return_value=mocked_u_shaped_model)
        mocker.patch("platypus.engine.prepare_loss_and_metrics", return_value=(None, None))
        self.initialized_engine.compile_u_shaped_model(model_cfg=model_cfg, segmentation_spec=self.config.semantic_segmentation)
        assert True

    def test_get_model_names(self):
        model_names = self.initialized_engine.get_model_names(config=dict(self.config), task="semantic_segmentation")
        assert model_names == ["model_name"]

