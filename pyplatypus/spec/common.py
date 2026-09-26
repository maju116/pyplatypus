"""Shared vocabulary for the spec.

Two rules from PLAN.md are enforced here rather than left to discipline:

* nothing hardcodes two spatial dimensions - `SpatialShape` accepts rank 2 or 3, and
  `rank` is derived from the data, never passed separately and never assumed;
* mutually exclusive options are an enum, not a pile of booleans. The old package had
  `resunet`, `linknet`, `plus_plus` as independent flags, which permits nonsense like
  a LinkNet that is also a Res-U-Net.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Reject unknown keys. A typo in a YAML key should be an error the user can see, not a
# setting that silently does nothing - that failure mode cost the old package real time.
# populate_by_name so a field with an alias can still be given by its real name from
# Python. Without it, `SegmentationData(window="lung")` would be rejected while the
# YAML key worked, which is the sort of asymmetry nobody can guess.
STRICT = ConfigDict(extra="forbid", frozen=True, validate_default=True,
                    populate_by_name=True)


class SpecModel(BaseModel):
    model_config = STRICT


SpatialShape = Annotated[
    tuple[int, ...],
    Field(min_length=2, max_length=3,
          description="Spatial size: (height, width) in 2D, (depth, height, width) in 3D."),
]


class Architecture(str, Enum):
    """One field, mutually exclusive by construction."""

    U_NET = "u_net"
    U_NET_PLUS_PLUS = "u_net_plus_plus"
    RES_U_NET = "res_u_net"
    LINKNET = "linknet"


class DataMode(str, Enum):
    NESTED_DIRS = "nested_dirs"
    CONFIG_FILE = "config_file"


class Activation(str, Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    GELU = "gelu"
    SILU = "silu"
    TANH = "tanh"



# The windows radiologists actually use, as (centre, width) in Hounsfield units. Naming
# one is clearer and less error-prone than typing two numbers, and it puts the vocabulary
# of the field into the API rather than leaving it in a paper somewhere. Kept here rather
# than beside the reader so that the spec stays free of numpy and pydicom.
WINDOWS: dict[str, tuple[float, float]] = {
    "brain": (40.0, 80.0),
    "subdural": (75.0, 215.0),
    "stroke": (32.0, 8.0),
    "bone": (400.0, 1800.0),
    "soft_tissue": (40.0, 400.0),
    "abdomen": (60.0, 400.0),
    "liver": (30.0, 150.0),
    "lung": (-600.0, 1500.0),
    "mediastinum": (50.0, 350.0),
}
