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
    shape_table,
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
    non_max_suppression,
)
from pyplatypus.detection.encode import (
    COCO_ANCHORS,
    STRIDES,
    Encoding,
    decode,
    encode,
    grid_shapes,
)
from pyplatypus.detection.loss import LossParts, Yolo3Loss
from pyplatypus.detection.metrics import (
    COCO_THRESHOLDS,
    DetectionError,
    DetectionMetrics,
    average_precision,
    detection_report,
    iou_matrix,
    match_detections,
)
from pyplatypus.detection.yolo3 import Darknet53, Yolo3, build_yolo3

__all__ = [
    "COCO_ANCHORS",
    "COCO_THRESHOLDS",
    "STRIDES",
    "AnchorFit",
    "Annotation",
    "Darknet53",
    "DetectionError",
    "DetectionMetrics",
    "Encoding",
    "Letterbox",
    "LossParts",
    "Yolo3",
    "Yolo3Loss",
    "anchor_coverage",
    "average_precision",
    "box_areas",
    "box_shapes",
    "build_yolo3",
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
    "non_max_suppression",
    "read_annotations",
    "read_labelme",
    "read_voc",
    "shape_table",
]
