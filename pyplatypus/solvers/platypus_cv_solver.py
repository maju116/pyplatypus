from pyplatypus.engine import PlatypusEngine
from pyplatypus.utils.config_processing_functions import YamlConfigLoader
from pathlib import Path


class PlatypusSolver(PlatypusEngine):
    """The wrapper around the mighty platypus engine which arms it with the tools for training
    multiple models tackling the CV tasks such as: semantic segmentation, object detections and
    image classification.
    """
    def __init__(self, config_yaml_path: str):
        """The initialization method offers loading the YAML config on the run with the use of validated config path.
        After the initialization we have global_cache and parsed config at hand, ready to be used.

        Parameters
        ----------
        config_yaml_path : str
            The path pointing at the YAML config to be used for the specific run.

        Raises
        ------
        NotADirectoryError
            In case of the config path being invalid.
        """
        if not Path(config_yaml_path).exists():
            raise NotADirectoryError(
                "The specified config path does not exist!"
                )
        else:
            ycl = YamlConfigLoader(config_yaml_path)
            self.config = ycl.load()
            super().__init__(self.config)

    def run(self):
        """For the time being the sole function of this method is to invoke the training mechanism but it is meant to be the tool
        for ordering the tasks and making the underlying actions clear for the user.
        """
        self.train()
        return None
