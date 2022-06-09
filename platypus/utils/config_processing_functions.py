from yaml import load, FullLoader
from pathlib import Path
from platypus.data_models.platypus_engine_datamodel import PlatypusSolverInput
from platypus.data_models.semantic_segmentation_datamodel import (
    SemanticSegmentationData, SemanticSegmentationInput, SemanticSegmentationModelSpec
    )
from platypus.data_models.object_detection_datamodel import ObjectDetectionInput
from platypus.data_models.augmentation_datamodel import (
    AugmentationSpecFull, ToFloatSpec, RandomRotate90Spec
    )  # TODO How to make it more generic?


class YamlConfigLoader(object):

    def __init__(self, config_yaml_path: str):
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

        Returns:
            Configuration (dict).
        """
        with open(config_path) as cfg:
            config = load(cfg, Loader=FullLoader)
        return config

    @staticmethod
    def create_semantic_segmentation_config(config: dict):
        data_ = SemanticSegmentationData(**config.get("semantic_segmentation").get("data"))
        models_ = [
            SemanticSegmentationModelSpec(**m) for m in config.get("semantic_segmentation").get("models")
            ]
        semantic_segmentation_ = SemanticSegmentationInput(data=data_, models=models_)
        return semantic_segmentation_

    def create_object_detection_config(config: dict):
        # Placeholder for now
        return ObjectDetectionInput()

    @staticmethod
    def create_augmentation_config(config: dict):
        tofloat_ = ToFloatSpec(**config.get("augmentation").get("ToFloat"))
        randomrotate_ = RandomRotate90Spec(**config.get("augmentation").get("RandomRotate90"))
        augmentation_ = AugmentationSpecFull(ToFloat=tofloat_, RandomRotate90=randomrotate_)
        return augmentation_

    def load(self):
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

    Args:
        config (dict): Configuration dictionary with tasks

    Returns:

    """
    available_tasks = ['object_detection', 'semantic_segmentation']
    return [task for task in config.keys() if config[task] is not None and task in available_tasks]
