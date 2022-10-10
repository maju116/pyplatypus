"""Script storing the main input for the solver, with the use of it, the data from YAML config gets parsed
through pydantic structures."""

from pydantic import BaseModel
from pyplatypus.data_models.object_detection_datamodel import ObjectDetectionInput
from pyplatypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationInput


class PlatypusSolverInput(BaseModel):
    object_detection: ObjectDetectionInput
    semantic_segmentation: SemanticSegmentationInput
