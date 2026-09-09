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
STRICT = ConfigDict(extra="forbid", frozen=True, validate_default=True)


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

