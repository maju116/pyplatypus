"""pyplatypus - the engine behind the platypus R package.

Specification, data pipeline, U-shaped models, losses and metrics, training, splitting by
patient and scoring per case, in 2D and 3D. Volumes arrive as NIfTI, reoriented to canonical
and windowed in real units; masks are colour pictures or label maps.

Three tasks, chosen by `task` in the specification: semantic segmentation through `Engine`,
object detection through `DetectionEngine`, and instance segmentation, whose specification
validates while its engine is still being built (`pyplatypus#160`). `build_engine(spec)`
returns the right one for the first two, so holding a spec is enough.
"""

from pyplatypus.data.images import read_image
from pyplatypus.data.masks import save_class_masks
from pyplatypus.data.splits import split_dataset, split_samples
from pyplatypus.detection_engine import DetectionEngine, DetectionReport, build_engine
from pyplatypus.engine import Engine, summarise_cases
from pyplatypus.errors import ConfigError, PlatypusError
from pyplatypus.instance_engine import InstanceEngine
from pyplatypus.plots import (
    flow_to_colours,
    overlay_agreement,
    overlay_instance_agreement,
    overlay_instances,
    overlay_mask,
    plot_anchors,
    plot_boxes,
    plot_masks,
)
from pyplatypus.runs import read_record, write_record
from pyplatypus.spec import (
    DetectionSpec,
    InstanceSpec,
    PlatypusSpec,
    SegmentationSpec,
    Separation,
    Task,
    available_callbacks,
    available_decodings,
    available_losses,
    available_metrics,
    available_models,
    available_optimizers,
    available_separations,
    available_tasks,
    available_transforms,
    from_dict,
    from_yaml,
    save_spec,
    spec_schema,
    write_schema,
)
from pyplatypus.style import drawing_style
from pyplatypus.weights import (
    WeightsError,
    available_weights,
    export_weights,
    resolve_weights,
)

__version__ = "0.8.0a26"
__all__ = [
    "ConfigError",
    "DetectionEngine",
    "DetectionReport",
    "DetectionSpec",
    "Engine",
    "InstanceEngine",
    "InstanceSpec",
    "PlatypusError",
    "PlatypusSpec",
    "SegmentationSpec",
    "Separation",
    "Task",
    "WeightsError",
    "__version__",
    "available_callbacks",
    "available_decodings",
    "available_losses",
    "available_metrics",
    "available_models",
    "available_optimizers",
    "available_separations",
    "available_tasks",
    "available_transforms",
    "available_weights",
    "build_engine",
    "drawing_style",
    "export_weights",
    "flow_to_colours",
    "from_dict",
    "from_yaml",
    "overlay_agreement",
    "overlay_instance_agreement",
    "overlay_instances",
    "overlay_mask",
    "plot_anchors",
    "plot_boxes",
    "plot_masks",
    "read_image",
    "read_record",
    "resolve_weights",
    "save_class_masks",
    "save_spec",
    "spec_schema",
    "split_dataset",
    "split_samples",
    "summarise_cases",
    "write_record",
    "write_schema",
]
