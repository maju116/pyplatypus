"""For the time being it is just a placeholder but with the new models flowing in, here the main config steering
te object detection models will be stored."""

from pydantic import BaseModel
from typing import Optional


class ObjectDetectionInput(BaseModel):
    type: Optional[str] = "__placeholder__"
