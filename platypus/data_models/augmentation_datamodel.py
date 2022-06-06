from pydantic import BaseModel
from typing import Optional


class ToFloatSpec(BaseModel):
    max_value: Optional[int] = 255
    always_apply: Optional[bool] = False
    p: Optional[float] = 1.


class RandomRotate90Spec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[float] = 1.


class AugmentationSpecFull(BaseModel):
    ToFloat: ToFloatSpec
    RandomRotate90: RandomRotate90Spec
