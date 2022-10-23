"""This module offers the tools crucial in the config ingestion but also it's latter validation
while some of the validation steps could get moved to the pydantic side but for now they are kept here.

Classes
-------
YamlConfigLoader(object)
    Provides the framework allowing us to ingest the raw config from the YAML shaped by a user.

Functions
---------
check_cv_tasks(config: dict)
    Checks which Computer Vision tasks are to be performed.
"""

from yaml import load, FullLoader
from pathlib import Path
from pyplatypus.utils.toolbox import convert_to_camel_case
from pyplatypus.data_models.platypus_engine_datamodel import PlatypusSolverInput
from pyplatypus.data_models.semantic_segmentation_datamodel import (
    SemanticSegmentationData, SemanticSegmentationInput, SemanticSegmentationModelSpec
    )
from pyplatypus.data_models.object_detection_datamodel import ObjectDetectionInput
from pyplatypus.data_models import augmentation_datamodel as AM
from pyplatypus.data_models import optimizer_datamodel as OM
from pyplatypus.data_models import callbacks_datamodel as CM
from pyplatypus.data_models import semantic_segmentation_loss_datamodel as SSLM

from pyplatypus.config.input_config import available_callbacks_without_specification


class YamlConfigLoader(object):
    """Provides the framework allowing us to ingest the raw config from the
    YAML shaped by a user. It provides us with the tool for directing the flow
    of data contained in the config through the branching PlatypusSolverInput
    structure. Therefore the essential information is extracted, validated, parsed
    and neatly organized in the expected structure.

    Methods
    -------
    load_config_from_yaml(config_path: str)
        Loads configuration from YAML file.

    create_semantic_segmentation_config(config: dict)
        Extracts the data regarding the semantic segmentation CV task to be performed.

    create_object_detection_config(config: dict)
        Extracts the data regarding the object detection CV task to be performed.

    create_augmentation_config(config: dict)
        Extracts the configuration later used during the augmentation pipeline and data generators creation.

    load()
        Loads the raw config from the YAML file, then each crucial configuration gets extracted from the main config
        and parsed through the Pydantic datamodel.
    """

    def __init__(self, config_yaml_path: str) -> None:
        """Initializes the class and checks the provided path to the YAML config.

        Parameters
        ----------
        config_yaml_path:
            Path to the user-specified config, it is required to exist.

        Raises
        ------
        NotADirectoryError: Raised if the path said to point at the config is invalid.
        """
        if not Path(config_yaml_path).exists():
            raise NotADirectoryError(
                "The specified config path does not exist!"
                )
        else:
            self.config_yaml_path = config_yaml_path

    @staticmethod
    def load_config_from_yaml(config_path: str) -> dict:
        """
        Loads configuration from YAML file.

        Parameters
        ----------
        config_path: str
            Path from which the YAML config will be loaded into the memory.

        Returns
        -------
        config: dict
            Loaded raw config.
        """
        with open(config_path) as cfg:
            config = load(cfg, Loader=FullLoader)
        return config

    @staticmethod
    def process_optimizer_field(model_config: dict) -> dict:
        """Extracts the information regarding optimizer.

        Parameters
        ----------
        model_config : dict
            The element of the list of models from the input config.

        Returns
        -------
        model_config: dict
            Reshaped config.
        """
        optimizer_field = model_config.get("optimizer")
        if optimizer_field is not None:
            optimizer_name = list(optimizer_field.keys())[0]
            optimizer_params = optimizer_field.get(optimizer_name)
            optimizer_spec = getattr(OM, f"{optimizer_name}Spec")(**optimizer_params)
            model_config.update(optimizer=optimizer_spec)
        if optimizer_field is None and "optimizer" in model_config.keys():
            model_config.pop("optimizer")
        return model_config

    @staticmethod
    def process_callbacks_field(model_config: dict) -> dict:
        """Extracts the information regarding callbacks and validates them prior to the latter pydantic-powered parsing.

        Parameters
        ----------
        model_config : dict
            The element of the list of models from the input config.

        Returns
        -------
        model_config: dict
            Reshaped config.

        Raises
        ------
        ValueError
            If the callbacks field is not recognized i.e. when it is not list nor dictionary.
        """
        callbacks_field = model_config.get("callbacks")
        if callbacks_field is not None:
            if isinstance(callbacks_field, list):
                callbacks = list(set(callbacks_field).intersection(set(available_callbacks_without_specification)))
                callbacks = [getattr(CM, f"{callback_name}Spec")() for callback_name in callbacks]
            elif isinstance(callbacks_field, dict):
                if "TerminateOnNaN" in callbacks_field.keys():
                    callbacks_field.update(TerminateOnNaN={})
                callbacks = [
                    getattr(
                        CM, f"{callback_name}Spec"
                        )(**callbacks_field.get(callback_name)) for callback_name in callbacks_field.keys()
                    ]
            else:
                raise ValueError("The structure of input not recognized for callback field!")
            model_config.update(callbacks=callbacks)
        if callbacks_field is None and "callbacks" in model_config.keys():
            model_config.pop("callbacks")
        return model_config

    @staticmethod
    def process_loss_field(model_config: dict) -> dict:
        """Transforms the loss field into pydantic data model, used later while defining the loss and metrics.

        Parameters
        ----------
        model_config : dict
            Configuration upon which the model is built.

        Returns
        -------
        model_config: dict
            Processed config.

        Raises
        ------
        ValueError
            If the field has unexpected format e.g. list.
        """
        loss_field = model_config.get("loss")
        if loss_field is not None:
            if isinstance(loss_field, str):
                cc_loss_name = convert_to_camel_case(loss_field)
                loss_spec = getattr(SSLM, f"{cc_loss_name}Spec")()
                model_config.update(loss=loss_spec)
            elif isinstance(loss_field, dict):
                loss_name = list(loss_field.keys())[0]
                loss_params = loss_field.get(loss_name) if loss_field.get(loss_name) else {}
                cc_loss_name = convert_to_camel_case(loss_name)
                loss_spec = getattr(SSLM, f"{cc_loss_name}Spec")(**loss_params)
                model_config.update(loss=loss_spec)
            else:
                raise ValueError("The structure of input not recognized for loss field!")
        if loss_field is None and "loss" in model_config.keys():
            model_config.pop("loss")
        return model_config

    @staticmethod
    def process_metrics_field(model_config: dict) -> dict:
        """Extracts the information regarding metrics and validates them prior to the latter pydantic-powered parsing.

        Parameters
        ----------
        model_config : dict
            The element of the list of models from the input config.

        Returns
        -------
        model_config: dict
            Reshaped config.

        Raises
        ------
        ValueError
            If the metrics field is not recognized i.e. when it is not list nor dictionary.
        """
        metrics_field = model_config.get("metrics")
        if metrics_field is not None:
            if isinstance(metrics_field, list):
                metrics_names = list(set(metrics_field))
                cc_metrics_names = [convert_to_camel_case(metric_name) for metric_name in metrics_names]
                metrics = [getattr(SSLM, f"{cc_metric_name}Spec")() for cc_metric_name in cc_metrics_names]
            elif isinstance(metrics_field, dict):
                cc_metrics_names = [convert_to_camel_case(metric_name) for metric_name in metrics_field.keys()]
                metrics = [
                    getattr(SSLM, f"{cc_metric_name}Spec")(**metrics_field.get(metric_name)) if metrics_field.get(
                        metric_name
                        ) else getattr(SSLM, f"{cc_metric_name}Spec")() for cc_metric_name, metric_name in zip(cc_metrics_names, metrics_field.keys())
                    ]
            else:
                raise ValueError("The structure of input not recognized for callback field!")
            model_config.update(metrics=metrics)
        if metrics_field is None and "metrics" in model_config.keys():
            model_config.pop("metrics")
        return model_config

    @staticmethod
    def process_augmentation_field(model_config: dict) -> dict:
        """Extracts the configuration later used during the augmentation pipeline and data generators creation.
        Then the data is parsed to the correct types with the use of predefined pydantic-based datamodels.

        Parameters
        ----------
        config: dict
            Configuration created by the user. The lacking fields will be filled with the default values.

        Returns
        -------
        augmentation_: dict
            Stores the arguments for every implemented transformation. The ones not defined in the user input
            are set to None and then filtered out."""
        augmentation_field = model_config.get("augmentation")
        if augmentation_field is not None:
            if isinstance(augmentation_field, list):
                augmentation = [getattr(AM, f"{callback_name}Spec")() for callback_name in augmentation_field]
            elif isinstance(augmentation_field, dict):
                augmentation = [
                    getattr(
                        AM, f"{augmentation_name}Spec"
                        )(**augmentation_field.get(augmentation_name)) for augmentation_name in augmentation_field.keys()
                    ]
            else:
                raise ValueError("The structure of input not recognized for callback field!")
            model_config.update(augmentation=augmentation)
        if augmentation_field is None and "augmentation" in model_config.keys():
            model_config.pop("augmentation")
        return model_config

    def create_semantic_segmentation_config(self, config: dict) -> SemanticSegmentationInput:
        """Extracts the data regarding the semantic segmentation CV task to be performed.
        Then the data is parsed to the correct types with the use of predefined pydantic-based datamodels.

        Parameters
        ----------
        config: dict
            Configuration created by the user. The lacking fields will be filled with the default values.

        Returns
        -------
        semantic_segmentation_: SemanticSegmentationInput
            Stores the fields controlling the training workflow e.g. the list of models that are to be trained."""
        data_ = SemanticSegmentationData(**config.get("semantic_segmentation").get("data"))
        models_ = []
        for m in config.get("semantic_segmentation").get("models"):
            m = self.process_optimizer_field(model_config=m)
            m = self.process_callbacks_field(model_config=m)
            m = self.process_loss_field(model_config=m)
            m = self.process_metrics_field(model_config=m)
            m = self.process_augmentation_field(model_config=m)
            models_.append(SemanticSegmentationModelSpec(**m))
        semantic_segmentation_ = SemanticSegmentationInput(data=data_, models=models_)
        return semantic_segmentation_

    @staticmethod
    def create_object_detection_config(config: dict) -> ObjectDetectionInput:
        """Extracts the data regarding the object detection CV task to be performed.
        Then the data is parsed to the correct types with the use of predefined pydantic-based datamodels.

        Parameters
        config: dict
            Configuration created by the user. The lacking fields will be filled with the default values.

        Returns
        -------
        object_detection_: ObjectDetectionInput
            Controlles the object detection workflow."""
        object_detection_ = ObjectDetectionInput()
        return object_detection_

    def load(self) -> PlatypusSolverInput:
        """Loads the raw config from the YAML file, then each crucial configuration gets extracted from the main config
        and parsed through the Pydantic datamodel. Then every component is used for the Platypus Solver input creation.

        Returns
        -------
        platypus_config: PlatypusSolverInput
            Complex object storing the configurations for object detection, semantic segmentation and augmentation
            thus controlling the whole workflow."""
        config_yaml_path = self.config_yaml_path
        raw_config = self.load_config_from_yaml(config_yaml_path)

        semantic_segmentation_ = self.create_semantic_segmentation_config(config=raw_config)
        object_detection_ = self.create_object_detection_config(config=raw_config)

        platypus_config = PlatypusSolverInput(
            object_detection=object_detection_,
            semantic_segmentation=semantic_segmentation_
            )
        return platypus_config


def check_cv_tasks(
    config: dict
        ) -> list:
    """
    Checks which Computer Vision tasks are to be performed.

    Parameters
    ----------
    config: dict
        Configuration dictionary with tasks.

    Returns
    -------
    selected_tasks: list
        Intersection of the selected and the available tasks.
    """
    available_tasks = ['object_detection', 'semantic_segmentation']
    selected_tasks = [task for task in config.keys() if config[task] is not None and task in available_tasks]
    return selected_tasks
