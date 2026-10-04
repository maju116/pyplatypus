"""pyplatypus - the engine behind the platypus R package.

Specification, data pipeline, U-shaped models, losses and metrics, training, splitting by
patient and scoring per case, in 2D and 3D. Volumes arrive as NIfTI, reoriented to canonical
and windowed in real units; masks are colour pictures or label maps.

Two tasks, chosen by `task` in the specification: segmentation through `Engine`, and object
detection through `DetectionEngine`. `build_engine(spec)` returns the right one, so holding
a spec is enough.
"""

from pyplatypus.data.splits import split_dataset, split_samples
from pyplatypus.detection_engine import DetectionEngine, DetectionReport, build_engine
from pyplatypus.engine import Engine, summarise_cases
from pyplatypus.errors import ConfigError, PlatypusError
from pyplatypus.runs import read_record, write_record
from pyplatypus.spec import (
    DetectionSpec,
    PlatypusSpec,
    SegmentationSpec,
    Task,
    from_dict,
    from_yaml,
    spec_schema,
    write_schema,
)
from pyplatypus.weights import (
    WeightsError,
    export_weights,
    known_weights,
    resolve_weights,
)

__version__ = "0.3.0a13"
__all__ = [
    "ConfigError",
    "DetectionEngine",
    "DetectionReport",
    "DetectionSpec",
    "Engine",
    "PlatypusError",
    "PlatypusSpec",
    "SegmentationSpec",
    "Task",
    "WeightsError",
    "__version__",
    "build_engine",
    "export_weights",
    "from_dict",
    "from_yaml",
    "known_weights",
    "read_record",
    "resolve_weights",
    "spec_schema",
    "split_dataset",
    "split_samples",
    "summarise_cases",
    "write_record",
    "write_schema",
]
