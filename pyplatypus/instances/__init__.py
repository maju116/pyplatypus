"""Instances: scoring first, and deliberately no model.

Detection worked here because it was built metrics-first - every piece checked against
something outside itself before anything was trained - and a model chosen before there is a
way to score it is a model chosen on nothing. So this package starts as a reader, a yardstick
post-process and the challenge's metric, and the first thing it produces is a number for the
weights that are already published.

The separation post-process (#159) arrived the same way and for the same reason: an interior
and a rim, grown back to instances, **measured against the ground truth before any model was
trained** - because the post-process is what caps the approach, and a cap can be measured
without the thing it caps. On the Data Science Bowl's 134 held-out cases that ceiling is an
instance AP of 0.9846 where the same truth through connected components reaches 0.7655.
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
from pyplatypus.instances.postprocess import (
    instances_from_mask,
    instances_from_parts,
    parts_from_instances,
)
from pyplatypus.instances.read import read_instance_masks, read_instances

__all__ = [
    "CHALLENGE_THRESHOLDS",
    "InstanceError",
    "average_precision",
    "instances_from_mask",
    "instances_from_parts",
    "iou_matrix",
    "label_map",
    "parts_from_instances",
    "read_instance_masks",
    "read_instances",
    "score_cases",
    "summarise_instances",
]
