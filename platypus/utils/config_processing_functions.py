from yaml import load, FullLoader
from pathlib import Path
from platypus.data_models.platypus_engine_datamodel import PlatypusSolverInput
from platypus.data_models.semantic_segmentation_datamodel import (
    SemanticSegmentationData, SemanticSegmentationInput, SemanticSegmentationModelSpec
    )
from platypus.data_models.object_detection_datamodel import ObjectDetectionInput
from platypus.data_models import augmentation_datamodel as AM


class YamlConfigLoader(object):
    """Provides the framework allowing us to ingest the raw config from the
    YAML shaped by an user. It provides us with the tool for directing the flow
    of data contained in the config through the branching PlatypusSolverInput
    structure. Therefore the essential information is extracted, validated, parsed
    and neatly organized in the expected structure.
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
    def load_config_from_yaml(
            config_path: str
    ) -> dict:
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
    def create_semantic_segmentation_config(config: dict) -> SemanticSegmentationInput:
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
        models_ = [
            SemanticSegmentationModelSpec(**m) for m in config.get("semantic_segmentation").get("models")
            ]
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

    @staticmethod
    def create_augmentation_config(config: dict) -> AM.AugmentationSpecFull:
        """Extracts the configuration later used during the augmentation pipeline and data generators creation.
        Then the data is parsed to the correct types with the use of predefined pydantic-based datamodels.

        Parameters
        ----------
        config: dict
            Configuration created by the user. The lacking fields will be filled with the default values.

        Returns
        -------
        augmentation_: AugmentationSpecFull
            Stores the arguments for every implemented transformation. The ones not defined in the user input
            are set to None and then filtered out."""
        augmentation_config_ = config.get("augmentation")
        augmentation_raw = dict()
        for transform in augmentation_config_.keys():
            spec = getattr(AM, f"{transform}Spec")(**dict(augmentation_config_.get(transform)))
            augmentation_raw.update({transform: spec})
        augmentation_ = AM.AugmentationSpecFull(**augmentation_raw)
        return augmentation_

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
        augmentation_ = self.create_augmentation_config(config=raw_config)

        platypus_config = PlatypusSolverInput(
            object_detection=object_detection_,
            semantic_segmentation=semantic_segmentation_,
            augmentation=augmentation_
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
