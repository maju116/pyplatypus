import pandas as pd

from pyplatypus.utils.config_processing_functions import check_cv_tasks
from pyplatypus.utils.augmentation_toolbox import prepare_augmentation_pipelines
from pyplatypus.segmentation.generator import prepare_data_generator, predict_from_generator
from pyplatypus.segmentation.models.u_shaped_models import u_shaped_model
from pyplatypus.data_models.platypus_engine_datamodel import PlatypusSolverInput
from pyplatypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationModelSpec, \
    SemanticSegmentationInput
from pyplatypus.utils.prepare_loss_metrics import prepare_loss_and_metrics, prepare_optimizer, prepare_callbacks_list

from pyplatypus.utils.toolbox import transform_probabilities_into_binaries, concatenate_binary_masks
from pyplatypus.utils.prediction_utils import save_masks
from albumentations import Compose
import numpy as np
from typing import Optional


class PlatypusEngine:
    """
    The architecture serving the purpose of creating the data pipeline, creating and calibrating the models with the use of it
    and latter predictions production.

    Methods
    -------
    train(self)
        Creates the augmentation pipeline based on the input config. Then the function performs
        the selected tasks e.g. semantic segmentation, which consists of compiling and fitting the model
        using the train and validation data generators created prior to the fitting.

    build_and_train_segmentation_models(
        self, train_augmentation_pipeline: Optional[Compose], validation_augmentation_pipeline: Optional[Compose]
        )
        Compiles and trains the U-Shaped architecture utilized in tackling the semantic segmentation task.

    compile_u_shaped_model(model_cfg: SemanticSegmentationModelSpec, segmentation_spec: SemanticSegmentationInput)
        Builds and compiles the U-shaped tensorflow model.

    produce_and_save_predicted_masks(self, model_name: Optional[str] = None)
        If the name parameter is set to None, then the outputs are produced for all the trained models.
        Otherwise, the model pointed at is used.

    produce_and_save_predicted_masks_for_model(self, model_name: str, custom_data_path: Optional[str] = None)
        For a certain model, function produces the prediction for a test data, or any data chosen by the user.
        Then these predictions are transformed into the binary masks and saved to the local files.

    predict_based_on_test_generator(self, model_name: str, custom_data_path: Optional[str] = None):
        Produces predictions based on the selected model and the data generator created on the course of building this model.

    get_model_names(config: dict, task: Optional[str] = "semantic_segmentation"):
        Extracts the names of all models related to the selected task.
    """

    def __init__(self, config: PlatypusSolverInput):
        """
        Performs Computer Vision tasks based on the certain data model, defined via the Pydantic parser.

        Parameters
        ----------
        config: PlatypusSolverInput
            Stores the specific task-related configs nested as its attributes e.g. to the models that are to
            be trained can be accessed as config.semantic_segmentation.models.
        """
        self.config = dict(config)
        self.cache = {}

    def update_cache(
            self, model_name: str, model: u_shaped_model, training_history: pd.DataFrame, model_specification: dict,
            task_type: str = "semantic_segmentation"
    ):
        """Stores the trained model in the cache, under its name defined by a user.

        Parameters
        ----------
        model_name : str
            Comes straight from the input config.
        model : u_shaped_model
            Tensorflow model.
        training_history: pd.DataFrame
            Training history.
        model_specification : dict
            Input model configuration.
        task_type: str
            Computer Vision task performed by the model.
        """
        self.cache.get(task_type).update({
            model_name: {"model": model, "training_history": training_history,
                         "model_specification": model_specification}
        })

    def train(self) -> None:
        """
        Creates the augmentation pipeline based on the input config. Then the function performs
        the selected tasks e.g. semantic segmentation, which consists of compiling and fitting the model
        using the train and validation data generators created prior to the fitting.
        """
        cv_tasks_to_perform = check_cv_tasks(self.config)
        train_augmentation_pipeline, validation_augmentation_pipeline = prepare_augmentation_pipelines(
            config=self.config)
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
            train_data_generator = prepare_data_generator(
                data=spec.data, model_cfg=model_cfg, augmentation_pipeline=train_augmentation_pipeline,
                path=spec.data.train_path, only_images=False, return_paths=False
            )
            validation_data_generator = prepare_data_generator(
                data=spec.data, model_cfg=model_cfg, augmentation_pipeline=validation_augmentation_pipeline,
                path=spec.data.validation_path, only_images=False, return_paths=False
            )
            model = self.compile_u_shaped_model(model_cfg)
            callbacks = prepare_callbacks_list(callbacks_specs=model_cfg.callbacks)
            training_history = model.fit(
                train_data_generator,
                epochs=model_cfg.epochs,
                steps_per_epoch=train_data_generator.steps_per_epoch,
                validation_data=validation_data_generator,
                validation_steps=validation_data_generator.steps_per_epoch,
                callbacks=callbacks
            )
            self.update_cache(
                model_name=model_cfg.name, model=model, training_history=pd.DataFrame(training_history.history),
                model_specification=dict(model_cfg)
            )

    @staticmethod
    def compile_u_shaped_model(model_cfg: SemanticSegmentationModelSpec):
        """Builds and compiles the U-shaped tensorflow model.

        Parameters
        ----------
        model_cfg : SemanticSegmentationModelSpec
            The model specification used for shaping the U-shaped architecture.
        """
        model = u_shaped_model(
            **dict(model_cfg)
        ).model
        ftp = model_cfg.fine_tuning_path
        if ftp is not None:
            model.load_weights(ftp)
            print("Weights loaded from " + ftp)
        training_loss, metrics = prepare_loss_and_metrics(
            loss=model_cfg.loss, metrics=model_cfg.metrics, n_class=model_cfg.n_class
        )
        optimizer = prepare_optimizer(optimizer=model_cfg.optimizer)
        model.compile(
            loss=training_loss,
            optimizer=optimizer,
            metrics=metrics
        )
        return model

    def produce_and_save_predicted_masks(self, model_name: Optional[str] = None,
                                         task_type: str = "semantic_segmentation"):
        """If the name parameter is set to None, then the outputs are produced for all the trained models.
        Otherwise, the model pointed at is used.

        Parameters
        ----------
        model_name : str
            Name of the model, should be consistent with the input config.
        task_type : Optional[str], optional
            Task of interest, by default "semantic_segmentation"
        """
        if model_name is None:
            model_names = self.get_model_names(config=self.config, task_type=task_type)
            for model_name in model_names:
                self.produce_and_save_predicted_masks_for_model(model_name, task_type)
        else:
            self.produce_and_save_predicted_masks_for_model(model_name, task_type)

    def produce_and_save_predicted_masks_for_model(
            self, model_name: str, custom_data_path: Optional[str] = None, task_type: str = "semantic_segmentation"
    ):
        """For a certain model, function produces the prediction for a test data, or any data chosen by the user.
        Then these predictions are transformed into the binary masks and saved to the local files.

        Parameters
        ----------
        model_name : str
            Name of the model to use, should be consistent with the input config.
        custom_data_path : Optional[str], optional
            If provided, the data is loaded from a custom source.
        task_type : Optional[str], optional
            Task of interest, by default "semantic_segmentation"
        """
        predictions, paths, colormap, mode = self.predict_based_on_test_generator(
            model_name, custom_data_path, task_type
        )
        image_masks = []
        for prediction in predictions:
            prediction_binary = transform_probabilities_into_binaries(prediction)
            prediction_mask = concatenate_binary_masks(binary_mask=prediction_binary, colormap=colormap)
            image_masks.append(prediction_mask)
        save_masks(image_masks, paths, model_name, mode)

    def predict_based_on_test_generator(
            self, model_name: str, custom_data_path: Optional[str] = None, task_type: str = "semantic_segmentation"
    ) -> tuple:
        """Produces predictions based on the selected model and the data generator created on the course of building this model.

        Parameters
        ----------
        model_name : str
            Name of the model to use, should be consistent with the input config.
        custom_data_path : Optional[str], optional
            If provided, the data is loaded from a custom source.
        task_type : Optional[str], optional
            Task of interest, by default "semantic_segmentation"

        Returns
        -------
        predictions: np.array
            Consists of the predictions for all the data yielded by the generator.
        paths: list
            Paths to the original images.
        colormap: List[Tuple[int, int, int]]
            Class color map.
        """
        spec = self.config[task_type]
        m = self.cache.get(task_type).get(model_name).get("model")
        _, validation_augmentation_pipeline = prepare_augmentation_pipelines(config=self.config)
        if custom_data_path is None:
            path = spec.data.validation_path
        else:
            path = custom_data_path
        idx = [cfg.name for cfg in self.config['semantic_segmentation'].models].index(model_name)
        g = prepare_data_generator(
            data=spec.data, model_cfg=self.config['semantic_segmentation'].models[idx],
            augmentation_pipeline=validation_augmentation_pipeline,
            path=path, only_images=True, return_paths=False
        )
        mode = g.mode
        colormap = g.colormap
        predictions, paths = predict_from_generator(model=m, generator=g)
        return predictions, paths, colormap, mode

    @staticmethod
    def get_model_names(config: dict, task_type: Optional[str] = "semantic_segmentation") -> list:
        """Extracts the names of all models related to the selected task.

        Parameters
        ----------
        config : dict
            It is expected to be of the same form as the input config.
        task_type : Optional[str], optional
            Task of interest, by default "semantic_segmentation"

        Returns
        -------
        model_names: list
            Names of the models associated with the chosen task.
        """
        model_names = [model_cfg.name for model_cfg in config.get(task_type).models]
        return model_names
