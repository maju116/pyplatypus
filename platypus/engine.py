from platypus.utils.config_processing_functions import check_cv_tasks
from platypus.utils.augmentation import create_augmentation_pipeline
from platypus.segmentation.generator import segmentation_generator
from platypus.segmentation.loss import segmentation_loss
from platypus.segmentation.models.u_shaped_models import u_shaped_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from platypus.data_models.platypus_engine_datamodel import PlatypusSolverInput
from platypus.utils.toolbox import convert_to_snake_case

from platypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationData, SemanticSegmentationModelSpec
from albumentations import Compose
from typing import Optional, Tuple


class platypus_engine:

    def __init__(
            self,
            config: PlatypusSolverInput,
            cache: dict
    ) -> None:
        """
        Performs Computer Vision tasks based on the certain data model, defined via the Pydantic parser.

        Args:
            config: PlatypusSolverInput
                Stores the specific task-related configs nested as its attributes e.g. to the models that are to
                be trained can be accessed as config.semantic_segmentation.models.
            cache: dict
                Dictionary that could be used as the handy storage for both outputs and intermediate results.
        """
        self.config = dict(config)
        self.cache = cache

    def prepare_loss_and_metrics(
        self, loss: str, metrics: list, n_class: int, background_index: Optional[int] = None
            ) -> tuple:
        """
        Returns the losses and metrics as the functions.

        Parameters
        ----------
        loss: str
            Name of any format.
        metrics: list
            Names (any case) of the metrics to be used in the training and validation performance assessment.
        n_class: int
            Indicates to how many classes each pixel could be classified i.e. the number of possible classes.
        background_index: int
            Used mainly by the remove_index method points to the layer storing the background probabilities.

        Returns
        -------
        training_loss: function
            Ready-to-use function calculating the loss given the classes' probabilities and the ground truth.
        metrics_to_apply: list
            List of functions serving as the trainining/validation sets evaluators.
        """
        training_loss = self.prepare_loss_function(
            loss=loss, n_class=n_class, background_index=background_index
            )
        metrics_to_apply = self.prepare_metrics(metrics, n_class, background_index)
        return training_loss, metrics_to_apply

    def prepare_metrics(
        self, metrics: list, n_class: int, background_index: Optional[int] = None
            ):
        """
        Returns the metrics as the functions.

        Parameters
        ----------
        metrics: list
            Names (any case) of the metrics to be used in the training and validation performance assessment.
        n_class: int
            Indicates to how many classes each pixel could be classified i.e. the number of possible classes.
        background_index: int
            Used mainly by the remove_index method points to the layer storing the background probabilities.

        Returns
        -------
        metrics_to_apply: list
            List of functions serving as the trainining/validation sets evaluators.
        """
        metrics_to_apply = ["categorical_crossentropy"]
        for metric in metrics:
            metric_function = self.prepare_loss_function(
                loss=metric, n_class=n_class, background_index=background_index
                )
            metrics_to_apply.append(metric_function)
        metrics_to_apply = list(set(metrics_to_apply))
        return metrics_to_apply

    @staticmethod
    def prepare_loss_function(
        loss: str, n_class: int, background_index: Optional[int] = None
            ):
        """
        Returns the ready-to-use loss function in the format expected by the Tensorflow. The function is
        extracted as the attribute of the segmentation_loss function.

        Parameters
        ----------
        loss: str
            Name of any format.
        n_class: int
            Indicates to how many classes each pixel could be classified i.e. the number of possible classes.
        background_index: int
            Used mainly by the remove_index method points to the layer storing the background probabilities.

        Returns
        -------
        loss_function: function
            Ready-to-use function calculating the loss given the classes' probabilities and the ground truth.
        """
        loss = convert_to_snake_case(any_case=loss)
        loss_function = getattr(
            segmentation_loss(n_class, background_index), loss
            )
        return loss_function

    @staticmethod
    def prepare_augmentation_pipelines(config: dict) -> Tuple[Compose]:
        """
        Prepares the pipelines consisting of the transforms taken from the albumentations
        module.

        Parameters
        ----------
        config: dict
            Config steering the workflow, it is checked for the presence of the "augmentation" key.

        Returns
        -------
        augmentation_pipelines: Tuple[albumentations.Compose]
            Composed of the augmentation pipelines.
        """
        if config.get("augmentation") is not None:
            train_augmentation_pipeline = create_augmentation_pipeline(
                augmentation_dict=dict(config.get("augmentation")),
                train=True
                )
            validation_augmentation_pipeline = create_augmentation_pipeline(
                dict(config.get("augmentation")), False
                )
        else:
            train_augmentation_pipeline = None
            validation_augmentation_pipeline = None
        pipelines = (train_augmentation_pipeline, validation_augmentation_pipeline)
        return pipelines

    @staticmethod
    def prepare_data_generators(
        data: SemanticSegmentationData, model_cfg: SemanticSegmentationModelSpec,
        train_augmentation_pipeline: Compose, validation_augmentation_pipeline: Compose
            ) -> tuple:
        generators = []
        for path, pipeline in zip(
            [data.train_path, data.validation_path], [train_augmentation_pipeline, validation_augmentation_pipeline]
                ):
            generator_ = segmentation_generator(
                path=path,
                mode=data.mode,
                colormap=data.colormap,
                only_images=False,
                net_h=model_cfg.net_h,
                net_w=model_cfg.net_w,
                h_splits=model_cfg.h_splits,
                w_splits=model_cfg.w_splits,
                grayscale=model_cfg.grayscale,
                augmentation_pipeline=pipeline,
                batch_size=model_cfg.batch_size,
                shuffle=data.shuffle,
                subdirs=data.subdirs,
                column_sep=data.column_sep
            )
            generators.append(generator_)
        generators = tuple(generators)
        return generators

    def train(self) -> None:
        """
        Creates the augmentation pipeline based on the input config. Then the function performs
        the selected tasks e.g. semantic segmentation, which consists of compiling and fitting the model
        using the train and validation data generators created prior to the fitting.
        """
        cv_tasks_to_perform = check_cv_tasks(self.config)
        train_augmentation_pipeline, validation_augmentation_pipeline = self.prepare_augmentation_pipelines(config=self.config)
        if 'semantic_segmentation' in cv_tasks_to_perform:
            spec = self.config['semantic_segmentation']
            for model_cfg in self.config['semantic_segmentation'].models:
                train_data_generator, validation_data_generator = self.prepare_data_generators(
                    data=spec.data, model_cfg=model_cfg, train_augmentation_pipeline=train_augmentation_pipeline,
                    validation_augmentation_pipeline=validation_augmentation_pipeline
                    )
                model = u_shaped_model(
                    **dict(model_cfg)
                ).model
                training_loss, metrics = self.prepare_loss_and_metrics(
                    loss=spec.data.loss, metrics=spec.data.metrics, n_class=model_cfg.n_class
                    )
                model.compile(
                    loss=training_loss,
                    optimizer=spec.data.optimizer.lower(),
                    metrics=metrics
                )
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
                        mode='max'  # TODO Is monitor supposed to be the str or our function?
                    ), EarlyStopping(
                        monitor='categorical_crossentropy', mode='max', patience=5
                    )]
                )
        return None
