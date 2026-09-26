"""pyplatypus - the engine behind the platypus R package.

Specification, data pipeline, U-shaped models, losses and metrics, training, splitting by
patient and scoring per case, in 2D and 3D. Volumes arrive as NIfTI, reoriented to canonical
and windowed in real units; masks are colour pictures or label maps.
"""

from pyplatypus.data.splits import split_dataset, split_samples
from pyplatypus.engine import Engine, summarise_cases
from pyplatypus.errors import ConfigError, PlatypusError
from pyplatypus.spec import PlatypusSpec, from_dict, from_yaml, spec_schema, write_schema

__version__ = "0.3.0a4"
__all__ = [
    "ConfigError",
    "Engine",
    "PlatypusError",
    "PlatypusSpec",
    "__version__",
    "from_dict",
    "from_yaml",
    "spec_schema",
    "split_dataset",
    "split_samples",
    "summarise_cases",
    "write_schema",
]
