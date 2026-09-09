"""pyplatypus - the engine behind the platypus R package."""

from pyplatypus.errors import ConfigError, PlatypusError
from pyplatypus.spec import PlatypusSpec, from_dict, from_yaml, spec_schema, write_schema

__version__ = "0.2.0.dev0"
__all__ = [
    "ConfigError",
    "PlatypusError",
    "PlatypusSpec",
    "__version__",
    "from_dict",
    "from_yaml",
    "spec_schema",
    "write_schema",
]
