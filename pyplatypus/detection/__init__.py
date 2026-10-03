"""Object detection: boxes rather than masks.

Metrics come first, deliberately. The package this replaces had a YOLOv3 that trained and
drew pictures and reported no precision, no recall and no average precision - so there was
no way to say whether it detected anything, and no number to put beside published weights.
Everything else here is unmeasurable until this part exists.
"""

from pyplatypus.detection.metrics import (
    COCO_THRESHOLDS,
    DetectionMetrics,
    average_precision,
    detection_report,
    iou_matrix,
    match_detections,
)

__all__ = [
    "COCO_THRESHOLDS",
    "DetectionMetrics",
    "average_precision",
    "detection_report",
    "iou_matrix",
    "match_detections",
]
