"""pyplatypus - the engine behind the platypus R package.

v0.1: specification, data pipeline, U-shaped models, losses and metrics, training.
2D segmentation. The spec and the model builder already handle volumes; the data
pipeline is where 3D stops for now.
"""

from pyplatypus.engine import Engine
from pyplatypus.errors import ConfigError, PlatypusError
from pyplatypus.spec import PlatypusSpec, from_dict, from_yaml, spec_schema, write_schema

__version__ = "0.2.0a2"
__all__ = [
    "ConfigError",
    "Engine",
    "PlatypusError",
    "PlatypusSpec",
    "__version__",
    "from_dict",
    "from_yaml",
    "spec_schema",
    "write_schema",
]
