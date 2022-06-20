from pydantic import BaseModel
from platypus.data_models.augmentation_datamodel import AugmentationSpecFull
from platypus.data_models.object_detection_datamodel import ObjectDetectionInput
from platypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationInput


class PlatypusSolverInput(BaseModel):
    object_detection: ObjectDetectionInput
    semantic_segmentation: SemanticSegmentationInput
    augmentation: AugmentationSpecFull
