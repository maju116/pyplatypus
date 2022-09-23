from platypus.utils.config_processing_functions import check_cv_tasks
from platypus.utils.augmentation_toolbox import prepare_augmentation_pipelines
from platypus.segmentation.generator import prepare_data_generators, segmentation_generator
from platypus.segmentation.models.u_shaped_models import u_shaped_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from platypus.data_models.platypus_engine_datamodel import PlatypusSolverInput
from platypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationModelSpec, SemanticSegmentationInput
from platypus.utils.prepare_loss_metrics import prepare_loss_and_metrics

from platypus.utils.toolbox import transform_probabilities_into_binaries, concatenate_binary_masks
from platypus.utils.prediction_utils import save_masks
from albumentations import Compose
import numpy as np
from typing import Optional


class platypus_engine:
    """
    The architecture serving the purpose of creating the data pipeline, creating and calibrating the models with the use of it
    and latter predictions production.

    Methods
    -------
    """
    def __init__(self, config: PlatypusSolverInput, cache: dict):
        """
        Performs Computer Vision tasks based on the certain data model, defined via the Pydantic parser.

        Parameters
        ----------
        config: PlatypusSolverInput
            Stores the specific task-related configs nested as its attributes e.g. to the models that are to
            be trained can be accessed as config.semantic_segmentation.models.
        cache: dict
            Dictionary that could be used as the handy storage for both outputs and intermediate results.
        """
        self.config = dict(config)
        self.cache = cache

    def update_cache(
        self, model_name: str, model: u_shaped_model, model_specification: dict, generator: segmentation_generator
            ):
        """Stores the trained model in the cache, under its name defined by an user.

        Parameters
        ----------
        model_name : str
            Comes straight from the input config.
        model : u_shaped_model
            Tensorflow model.
        model_specification : dict
            Input model configuration.
        generator: segmentation_generator
            Test generator.
        """
        self.cache.get("semantic_segmentation").update({
            model_name: {"model": model, "model_specification": model_specification, "data_generator": generator}
            })

    def train(self) -> None:
        """
        Creates the augmentation pipeline based on the input config. Then the function performs
        the selected tasks e.g. semantic segmentation, which consists of compiling and fitting the model
        using the train and validation data generators created prior to the fitting.
        """
        cv_tasks_to_perform = check_cv_tasks(self.config)
        train_augmentation_pipeline, validation_augmentation_pipeline = prepare_augmentation_pipelines(config=self.config)
        if 'semantic_segmentation' in cv_tasks_to_perform:
            self.cache.update(semantic_segmentation={})
            self.build_and_train_segmentation_models(train_augmentation_pipeline, validation_augmentation_pipeline)

    def build_and_train_segmentation_models(
        self, train_augmentation_pipeline: Optional[Compose], validation_augmentation_pipeline: Optional[Compose]
            ):
        """Compiles and trains the U-Shaped architecture utilized in tackling the semantic segmentation task.

        Parameters
        ----------
        train_augmentation_pipeline : Optional[Compose]
            Optional augmentation pipeline, native to the albumentations package.
        validation_augmentation_pipeline : Optional[Compose]
            Optional augmentation pipeline, native to the albumentations package.
        """
        spec = self.config['semantic_segmentation']
        for model_cfg in self.config['semantic_segmentation'].models:
            train_data_generator, validation_data_generator, test_data_generator = prepare_data_generators(
                data=spec.data, model_cfg=model_cfg, train_augmentation_pipeline=train_augmentation_pipeline,
                validation_augmentation_pipeline=validation_augmentation_pipeline
                )
            model = self.compile_u_shaped_model(model_cfg, segmentation_spec=spec)
            model.fit(
                train_data_generator,
                epochs=model_cfg.epochs,
                steps_per_epoch=train_data_generator.steps_per_epoch,
                validation_data=validation_data_generator,
                validation_steps=validation_data_generator.steps_per_epoch,
                callbacks=[ModelCheckpoint(
                    filepath=model_cfg.name + '.hdf5',
                    save_best_only=True,
                    monitor='categorical_crossentropy',  # TODO Add the monitor function and check if it is one of the metrics
                    mode='min'  # TODO Is monitor supposed to be the str or our function?
                ), EarlyStopping(
                    monitor='val_iou_coefficient', mode='max', patience=5
                )]
            )
            self.update_cache(
                model_name=model_cfg.name, model=model, model_specification=dict(model_cfg), generator=test_data_generator
                )

    def compile_u_shaped_model(model_cfg: SemanticSegmentationModelSpec, segmentation_spec: SemanticSegmentationInput):
        """Builds and compiles the U-shaped tensorflow model.

        Parameters
        ----------
        model_cfg : SemanticSegmentationModelSpec
            The model specification used for shaping the U-shaped architecture.
        segmentation_spec : SemanticSegmentationInput
            From here information such as loss function and metrics are taken.
        """
        model = u_shaped_model(
            **dict(model_cfg)
        ).model
        training_loss, metrics = prepare_loss_and_metrics(
            loss=segmentation_spec.data.loss, metrics=segmentation_spec.data.metrics, n_class=model_cfg.n_class
            )
        model.compile(
            loss=training_loss,
            optimizer=segmentation_spec.data.optimizer.lower(),
            metrics=metrics
        )

    def produce_and_save_predicted_masks(self, model_name):
        if model_name is None:
            model_names = self.get_model_names(config=self.config)
            for model_name in model_names:
                self.produce_and_save_predicted_masks_for_model(model_name)
        else:
            self.produce_and_save_predicted_masks_for_model(model_name)

    def produce_and_save_predicted_masks_for_model(self, model_name, custom_data_path=None):
        predictions, paths, colormap = self.predict_based_on_test_generator(model_name, custom_data_path)
        image_masks = []
        for prediction in predictions:
            prediction_binary = transform_probabilities_into_binaries(prediction)
            prediction_mask = concatenate_binary_masks(binary_mask=prediction_binary, colormap=colormap)
            image_masks.append(prediction_mask)
        save_masks(image_masks, paths, model_name)

    def predict_based_on_test_generator(self, model_name: str, custom_data_path: str):
        m = self.cache.get("semantic_segmentation").get(model_name).get("model")
        g = self.cache.get("semantic_segmentation").get(model_name).get("data_generator")
        if custom_data_path is not None:
            g.path = custom_data_path
        colormap = g.colormap
        predictions, paths = self.predict_from_generator(model=m, generator=g)
        return predictions, paths, colormap

    @staticmethod
    def get_model_names(config: dict, task: str = "semantic_segmentation"):
        model_names = [model_cfg.name for model_cfg in config.get(task).models]
        return model_names
