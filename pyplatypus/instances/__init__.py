"""Instances: scoring first, and deliberately no model.

Detection worked here because it was built metrics-first - every piece checked against
something outside itself before anything was trained - and a model chosen before there is a
way to score it is a model chosen on nothing. So this package starts as a reader, a yardstick
post-process and the challenge's metric, and the first thing it produces is a number for the
weights that are already published.
"""

from pyplatypus.instances.metrics import (
    CHALLENGE_THRESHOLDS,
    InstanceError,
    average_precision,
    iou_matrix,
    label_map,
    score_cases,
    summarise_instances,
)
from pyplatypus.instances.postprocess import instances_from_mask
from pyplatypus.instances.read import read_instance_masks, read_instances

__all__ = [
    "CHALLENGE_THRESHOLDS",
    "InstanceError",
    "average_precision",
    "instances_from_mask",
    "iou_matrix",
    "label_map",
    "read_instance_masks",
    "read_instances",
    "score_cases",
    "summarise_instances",
]
