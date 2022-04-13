from platypus.utils.config import load_config_from_yaml, check_tasks
from platypus.utils.augmentation import create_augmentation_pipeline
import numpy as np
import platypus.segmentation as seg
import platypus.detection as det


class platypus_engine:

    def __init__(
            self,
            config_yaml_path: str
    ) -> None:
        """
        Performs Computer Vision tasks based on YAML config file.

        Args:
            config_yaml_path (str): Path to the config YAML file.
        """
        self.config = load_config_from_yaml(config_path=config_yaml_path)
