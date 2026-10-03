"""Object detection: boxes rather than masks.

Metrics come first, deliberately. The package this replaces had a YOLOv3 that trained and
drew pictures and reported no precision, no recall and no average precision - so there was
no way to say whether it detected anything, and no number to put beside published weights.
Everything else here is unmeasurable until this part exists.
"""

from pyplatypus.detection.anchors import (
    AnchorFit,
    anchor_coverage,
    box_shapes,
    fit_shapes,
    generate_anchors,
)
from pyplatypus.detection.annotations import (
    Annotation,
    describe_annotations,
    read_annotations,
    read_labelme,
    read_voc,
)
from pyplatypus.detection.boxes import (
    Letterbox,
    box_areas,
    clip_boxes,
    drop_degenerate,
)
from pyplatypus.detection.encode import (
    COCO_ANCHORS,
    STRIDES,
    Encoding,
    decode,
    encode,
    grid_shapes,
)
from pyplatypus.detection.metrics import (
    COCO_THRESHOLDS,
    DetectionError,
    DetectionMetrics,
    average_precision,
    detection_report,
    iou_matrix,
    match_detections,
)

__all__ = [
    "COCO_ANCHORS",
    "COCO_THRESHOLDS",
    "STRIDES",
    "AnchorFit",
    "Annotation",
    "DetectionError",
    "DetectionMetrics",
    "Encoding",
    "Letterbox",
    "anchor_coverage",
    "average_precision",
    "box_areas",
    "box_shapes",
    "clip_boxes",
    "decode",
    "describe_annotations",
    "detection_report",
    "drop_degenerate",
    "encode",
    "fit_shapes",
    "generate_anchors",
    "grid_shapes",
    "iou_matrix",
    "match_detections",
    "read_annotations",
    "read_labelme",
    "read_voc",
]
