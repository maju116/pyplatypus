from platypus.engine import platypus_engine
from platypus.data_models.platypus_engine_datamodel import PlatypusSolverInput
from platypus.utils.config_processing_functions import YamlConfigLoader
from pathlib import Path

from typing import Optional


class PlatypusSolver(platypus_engine):

    def __init__(self, config_yaml_path: str, config: Optional[dict] = None):
        if not Path(config_yaml_path).exists():
            raise NotADirectoryError(
                "The specified config path does not exist while the already loaded config was not provided!"
                )
        else:
            ycl = YamlConfigLoader(config_yaml_path)
            self.config = ycl.load()
            self.cache = dict()
            super().__init__(self.config, self.cache)

    def run(self):
        self.train()
        return None
