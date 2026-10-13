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

`flows` (#212) is the second separation signal and was written in the same order, which is
now the pattern rather than a one-off: the target and its decoding first, their ceiling
measured on the ground truth, and only then a training run. It exists because `#159` closed
measured-and-no for a structural reason - an interior and a rim are two classes competing for
the same pixels - and a flow field spends no pixel on being a boundary.
"""

from pyplatypus.instances.decode import (
    Reading,
    decode_instances,
    read_prediction,
)
from pyplatypus.instances.flows import (
    FLOW_MODES,
    flows_from_instances,
    follow_flows,
    instances_by_flows,
)
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
    instances_by_watershed,
    instances_from_mask,
    instances_from_parts,
    parts_from_instances,
)
from pyplatypus.instances.read import read_instance_masks, read_instances
from pyplatypus.instances.targets import instance_target, resize_labels

__all__ = [
    "CHALLENGE_THRESHOLDS",
    "FLOW_MODES",
    "InstanceError",
    "Reading",
    "average_precision",
    "decode_instances",
    "flows_from_instances",
    "follow_flows",
    "instance_target",
    "instances_by_flows",
    "instances_by_watershed",
    "instances_from_mask",
    "instances_from_parts",
    "iou_matrix",
    "label_map",
    "parts_from_instances",
    "read_instance_masks",
    "read_instances",
    "read_prediction",
    "resize_labels",
    "score_cases",
    "summarise_instances",
]
