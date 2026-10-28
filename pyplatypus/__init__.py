"""pyplatypus - the engine behind the platypus R package.

Specification, data pipeline, U-shaped models, losses and metrics, training, splitting by
patient and scoring per case, in 2D and 3D. Volumes arrive as NIfTI, reoriented to canonical
and windowed in real units; masks are colour pictures or label maps.

Three tasks are chosen by `task` in the specification: semantic segmentation through
`Engine`, object detection through `DetectionEngine` and instance segmentation through
`InstanceEngine`. `build_engine(spec)` returns the right one for any of them, so holding a
spec is enough.

This paragraph said instance segmentation's engine was *still being built* until
2026-10-27, naming `pyplatypus#160` - which closed on 2026-10-13 as the spec tag **and**
the engine, and `build_engine` has returned an `InstanceEngine` ever since. Two weeks of
rot in the one docstring that is a package's front door, which is §4 of the workbench's
notes in its plainest form.

A fourth is being built: **classification**, one label per image, through
`ClassificationEngine`. It has no `task` tag yet and so no top-level specification - the tag
lands last, after everything behind it works, so that it never names a task this package
cannot train - which means the engine takes a `ClassificationData` and a list of
`ClassificationModel` rather than a spec, and `build_engine` does not know it. The same
state detection's own engine describes for a detector today.
"""

from pyplatypus.classification_engine import ClassificationEngine, ClassificationRun
from pyplatypus.data.images import read_image
from pyplatypus.data.masks import save_class_masks
from pyplatypus.data.splits import split_dataset, split_samples, stratified_split
from pyplatypus.data.volumes import save_class_volumes, save_volumes
from pyplatypus.detection_engine import DetectionEngine, DetectionReport, build_engine
from pyplatypus.device import device_report
from pyplatypus.engine import Engine, summarise_cases
from pyplatypus.errors import ConfigError, PlatypusError
from pyplatypus.instance_engine import InstanceEngine
from pyplatypus.measure import (
    ellipse_axes,
    ellipse_circumference,
    instance_shapes,
    instance_sizes,
    largest_filled_object,
    mask_volume,
)
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
    ClassificationSpec,
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

__version__ = "0.8.0a51"
__all__ = [
    "ClassificationEngine",
    "ClassificationRun",
    "ClassificationSpec",
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
    "device_report",
    "drawing_style",
    "ellipse_axes",
    "ellipse_circumference",
    "export_weights",
    "flow_to_colours",
    "from_dict",
    "from_yaml",
    "instance_shapes",
    "instance_sizes",
    "largest_filled_object",
    "mask_volume",
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
    "save_class_volumes",
    "save_spec",
    "save_volumes",
    "spec_schema",
    "split_dataset",
    "split_samples",
    "stratified_split",
    "summarise_cases",
    "write_record",
    "write_schema",
]
