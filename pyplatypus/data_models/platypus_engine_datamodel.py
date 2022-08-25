from pydantic import BaseModel
from pyplatypus.data_models.augmentation_datamodel import AugmentationSpecFull
from pyplatypus.data_models.object_detection_datamodel import ObjectDetectionInput
from pyplatypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationInput


class PlatypusSolverInput(BaseModel):
    object_detection: ObjectDetectionInput
    semantic_segmentation: SemanticSegmentationInput
    augmentation: AugmentationSpecFull
