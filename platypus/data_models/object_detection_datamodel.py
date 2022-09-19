from pydantic import BaseModel
from typing import Optional


class ObjectDetectionInput(BaseModel):
    type: Optional[str] = "__placeholder__"
