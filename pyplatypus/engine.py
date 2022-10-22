import pandas as pd
from random import randrange

from pandas import DataFrame, Series

from pyplatypus.utils.config_processing_functions import check_cv_tasks
from pyplatypus.utils.augmentation_toolbox import prepare_augmentation_pipelines
from pyplatypus.segmentation.generator import prepare_data_generator, predict_from_generator
from pyplatypus.segmentation.models.u_shaped_models import u_shaped_model
from pyplatypus.data_models.platypus_engine_datamodel import PlatypusSolverInput
from pyplatypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationModelSpec
from pyplatypus.utils.prepare_loss_metrics import prepare_loss_and_metrics, prepare_optimizer, prepare_callbacks_list

from pyplatypus.utils.toolbox import transform_probabilities_into_binaries, concatenate_binary_masks
from pyplatypus.utils.prediction_utils import save_masks
from albumentations import Compose
import numpy as np
from typing import Optional, Union


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
            self, model_name: str, model: u_shaped_model, training_history: pd.DataFrame,
            model_specification: SemanticSegmentationModelSpec, task_type: str = "semantic_segmentation"
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
        if 'semantic_segmentation' in cv_tasks_to_perform:
            self.cache.update(semantic_segmentation={})
            self.build_and_train_segmentation_models()

    def build_and_train_segmentation_models(
        self
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
            train_augmentation_pipeline, validation_augmentation_pipeline = prepare_augmentation_pipelines(config=model_cfg)
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
            if model_cfg.fit:
                training_history = model.fit(
                    train_data_generator,
                    epochs=model_cfg.epochs,
                    steps_per_epoch=train_data_generator.steps_per_epoch,
                    validation_data=validation_data_generator,
                    validation_steps=validation_data_generator.steps_per_epoch,
                    callbacks=callbacks
                )
                training_history = pd.DataFrame(training_history.history)
                best_model = self.serve_best_model(model, callbacks)
            else:
                training_history = pd.DataFrame()
                best_model = model
            self.update_cache(
                model_name=model_cfg.name, model=best_model, training_history=training_history,
                model_specification=model_cfg
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

    def produce_and_save_predicted_masks(self, custom_data_path: Optional[str] = None,
                                         task_type: str = "semantic_segmentation"):
        """If the name parameter is set to None, then the outputs are produced for all the trained models.
        Otherwise, the model pointed at is used.

        Parameters
        ----------
        custom_data_path : Optional[str], optional
            If provided, the data is loaded from a custom source.
        task_type : Optional[str], optional
            Task of interest, by default "semantic_segmentation"
        """
        model_names = self.get_model_names(config=self.config, task_type=task_type)
        for model_name in model_names:
            self.produce_and_save_predicted_masks_for_model(model_name, custom_data_path, task_type)

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
        predictions, paths, colormap, mode = self.predict_based_on_generator(
            model_name, custom_data_path, task_type
        )
        image_masks = []
        for prediction in predictions:
            prediction_binary = transform_probabilities_into_binaries(prediction)
            prediction_mask = concatenate_binary_masks(binary_mask=prediction_binary, colormap=colormap)
            image_masks.append(prediction_mask)
        save_masks(image_masks, paths, model_name, mode)

    def predict_based_on_generator(
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
        model_cfg = self.cache.get(task_type).get(model_name).get("model_specification")

        _, validation_augmentation_pipeline = prepare_augmentation_pipelines(config=model_cfg)
        if custom_data_path is None:
            path = spec.data.validation_path
        else:
            path = custom_data_path
        g = prepare_data_generator(
            data=spec.data, model_cfg=model_cfg,
            augmentation_pipeline=validation_augmentation_pipeline,
            path=path, only_images=True, return_paths=True
        )
        mode = g.mode
        colormap = g.colormap
        predictions, paths = predict_from_generator(model=m, generator=g)
        return predictions, paths, colormap, mode

    def sample_generators(self, custom_data_path: Optional[str] = None, training_augmentation: bool = True,
                          task_type: str = "semantic_segmentation"):
        """If the name parameter is set to None, then the outputs are produced for all the trained models.
        Otherwise, the model pointed at is used.

        Parameters
        ----------
        custom_data_path : Optional[str], optional
            If provided, the data is loaded from a custom source.
        training_augmentation: bool
            Should training or validation augmentation pipeline be used.
        task_type : Optional[str], optional
            Task of interest, by default "semantic_segmentation"
        """
        batches = []
        model_names = self.get_model_names(config=self.config, task_type=task_type)
        for model_name in model_names:
            batch = self.sample_generator(model_name, custom_data_path, training_augmentation, task_type)
            batches.append(batch)
        return batches

    def sample_generator(
            self, model_name: str, custom_data_path: Optional[str] = None, training_augmentation: bool = True,
            task_type: str = "semantic_segmentation"
    ) -> tuple:
        """Produces predictions based on the selected model and the data generator created on the course of building this model.

        Parameters
        ----------
        model_name : str
            Name of the model to use, should be consistent with the input config.
        custom_data_path : Optional[str], optional
            If provided, the data is loaded from a custom source.
        training_augmentation: bool
            Should training or validation augmentation pipeline be used.
        task_type : Optional[str], optional
            Task of interest, by default "semantic_segmentation"

        Returns
        -------
        batch: np.array
            Batch of augmentetd images.
        """
        spec = self.config[task_type]
        model_cfg = self.cache.get(task_type).get(model_name).get("model_specification")

        if training_augmentation:
            augmentation_pipeline, _ = prepare_augmentation_pipelines(config=model_cfg)
        else:
            _, augmentation_pipeline = prepare_augmentation_pipelines(config=model_cfg)
        if custom_data_path is None:
            path = spec.data.validation_path
        else:
            path = custom_data_path
        g = prepare_data_generator(
            data=spec.data, model_cfg=model_cfg,
            augmentation_pipeline=augmentation_pipeline,
            path=path, only_images=False, return_paths=False
        )
        batch = g.__getitem__(randrange(g.steps_per_epoch))
        return batch

    def evaluate_models(
            self, custom_data_path: str = None, task_type: str = "semantic_segmentation"
    ) -> Union[DataFrame, Series]:
        """Evaluates all the models associated with a certain task or the one specified by the model_name.

        Parameters
        custom_data_path : str, optional
            Makes evaluating on a data different from the one used for validation possible, by default None
        task_type : str, optional
            Task of interest, by default "semantic_segmentation"

        Returns
        -------
        evaluations: list
            List of DataFrames.
        """
        evaluations = []
        model_names = self.get_model_names(config=self.config, task_type=task_type)
        for model_name in model_names:
            prepared_evaluation_metrics = self.evaluate_model(model_name, custom_data_path, task_type)
            evaluations.append(prepared_evaluation_metrics)
        evaluations = pd.concat(evaluations)
        print("EVALUATION RESULTS:\n")
        print(evaluations)
        return evaluations

    def evaluate_model(
            self, model_name: str, custom_data_path: str = None, task_type: str = "semantic_segmentation"
    ) -> pd.DataFrame:
        """Prepares the crucial objects and evaluates model invoking the method calling the .evaluate() method
        with the use of validation generator.

        Parameters
        ----------
        model_name : str
            The model that is to be evaluated.
        custom_data_path : str, optional
            Makes evaluating on a data different from the one used for validation possible, by default None
        task_type : str
            Task with which the model is associated.

        Returns
        -------
        prepared_evaluation_metrics: pd.DataFrame
            The filled-in evaluation table.
        """
        task_cfg = self.cache.get(task_type)
        model_cfg = task_cfg.get(model_name).get("model_specification")
        evaluation_table = self.prepare_evaluation_table(model_cfg)
        evaluation_metrics = self.evaluate_based_on_generator(model_name, model_cfg, task_type, custom_data_path)
        prepared_evaluation_metrics = self.prepare_evaluation_results(
            evaluation_metrics, model_name, evaluation_columns=evaluation_table.columns
        )
        return prepared_evaluation_metrics

    @staticmethod
    def prepare_evaluation_table(model_cfg: dict) -> pd.DataFrame:
        """Creates empty table with the proper columns, to be filled during the evaluation, also from it the columns' names are taken
        to be used by other methods.

        Parameters
        ----------
        model_cfg : dict
            Dictionary that was used to define the model.

        Returns
        -------
        evaluation_table: pd.DataFrame
            Template table.
        """
        loss_name, metrics_names = model_cfg.loss.name, [metric.name for metric in model_cfg.metrics]
        evaluation_columns = ["model_name", loss_name, "categorical_crossentropy"] + metrics_names
        evaluation_table = pd.DataFrame(columns=evaluation_columns)
        return evaluation_table

    @staticmethod
    def prepare_evaluation_results(evaluation_metrics: list, model_name: str, evaluation_columns: list) -> pd.DataFrame:
        """Composes the data frame containing the model's and metrics' names alongside their values.

        Parameters
        ----------
        evaluation_metrics : list
            Metrics values, expected to be returned by a model's 'evaluate' method.
        model_name : str
            Name of the model.
        evaluation_columns : list
            Names of the loss function and metrics, extracted from the configuration file.

        Returns
        -------
        prepared_evaluation_metrics: pd.DataFrame
            Dataframe summarizing the run.
        """
        evaluation_results = [[model_name] + evaluation_metrics]
        prepared_evaluation_metrics = pd.DataFrame(evaluation_results, columns=evaluation_columns)
        return prepared_evaluation_metrics

    def evaluate_based_on_generator(
            self, model_name: str, model_cfg: SemanticSegmentationModelSpec, task_type: str = "semantic_segmentation",
            custom_data_path: Optional[str] = None
    ) -> tuple:
        """Produces metrics and loss value based on the selected model and the data generator created on the course
        of building this model.

        Parameters
        ----------
        model_name : str
            Name of the model to use, should be consistent with the input config.
        custom_data_path : Optional[str], optional
            If provided, the data is loaded from a custom source.
        model_cfg: SemanticSegmentationModelSpec
            Model configuration.
        task_type : str
            Task with which the model is associated.

        Returns
        -------
        metrics: np.array
            Consists of the predictions for all the data yielded by the generator.
        """
        task_cfg = self.config.get(task_type)
        m = self.cache.get(task_type).get(model_name).get("model")
        model_cfg = self.cache.get(task_type).get(model_name).get("model_specification")
        _, validation_augmentation_pipeline = prepare_augmentation_pipelines(config=model_cfg)
        if custom_data_path is None:
            path = task_cfg.data.validation_path
        else:
            path = custom_data_path
        g = prepare_data_generator(
            data=task_cfg.data, model_cfg=model_cfg,
            augmentation_pipeline=validation_augmentation_pipeline,
            path=path, only_images=False, return_paths=False
        )
        metrics = m.evaluate(x=g)
        return metrics

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

    @staticmethod
    def serve_best_model(model, callbacks: list):
        """Takes care of the case of using both EarlyStopping and ModelCheckopint when the last weights setting
        is not necessarily going to be the optimal one because of training interuption.
        It returns an input model or the one with the weights loaded from a checkpoint.

        Parameters
        ----------
        model : keras.engine.functional.Functional
            Compiled model.
        callbacks : list
            List of callbacks data models.

        Returns
        -------
        best_model: keras.engine.functional.Functional
            Model brought back using the checkpoint.
        """
        callback_names = [callback.name for callback in callbacks]
        if {"EarlyStopping", "ModelCheckpoint"}.issubset(set(callback_names)):
            model_checkpoint_callback = [callback for callback in callbacks if callback.name == "ModelCheckpoint"][0]
            model.load_weights(model_checkpoint_callback.filepath)
        return model
