from pydantic import BaseModel, validator
from pydantic import PositiveInt, conint, conlist, confloat
from typing import List, Optional, Union, Tuple
from pathlib import Path

from platypus.utils.toolbox import convert_to_snake_case
from platypus.config.input_config import implemented_models, implemented_modes, implemented_losses, implemented_metrics


class SemanticSegmentationData(BaseModel):
    train_path: str
    validation_path: str
    test_path: str
    colormap: Union[
        List[List[conint(ge=0, le=255)]],
        List[Tuple[conint(ge=0, le=255)]]
        ]
    mode: str
    shuffle: bool
    subdirs: conlist(str, min_items=2, max_items=2)
    column_sep: str
    loss: Optional[str] = "Iou loss"
    metrics: Optional[List[str]] = []

    @validator('train_path')
    def check_if_train_path_exists(cls, v: str):
        if Path(v).exists():
            return v
        raise NotADirectoryError("Specified train path does not exist!")

    @validator('validation_path')
    def check_if_validation_path_exists(cls, v: str):
        if Path(v).exists():
            return v
        raise NotADirectoryError("Specified validation path does not exist!")

    @validator('test_path')
    def check_if_test_path_exists(cls, v: str):
        if Path(v).exists():
            return v
        raise NotADirectoryError("Specified test path does not exist!")

    @validator("colormap")
    def check_colormanp_length(cls, v: list):
        if all([len(c) == 3 for c in v]):
            return v
        raise ValueError("The colormap must consist of three-element lists or tuples!")

    @validator('mode')
    def check_mode_value(cls, v: str):
        if v in implemented_modes:
            return v
        raise ValueError(f"Mode is supposed to be one of: {', '.join(implemented_modes)}!")

    @validator("loss")
    def check_the_loss_name(cls, v: str):
        if convert_to_snake_case(v) in implemented_losses:
            return v
        raise ValueError(f"The chosen loss: {v} is not one of the implemented losses!")

    @validator("metrics")
    def check_the_metrics(cls, v: list):
        v_converted = [convert_to_snake_case(case) for case in v]
        if set(v_converted).issubset(set(implemented_metrics)):
            return v
        raise ValueError(f"The chosen metrics: {', '.join(v)} are not the subset of the implemented ones!")

class SemanticSegmentationModelSpec(BaseModel):
    name: str
    type: str
    net_h: PositiveInt
    net_w: PositiveInt
    blocks: PositiveInt
    n_class: conint(ge=2)
    filters: PositiveInt
    dropout: confloat(ge=0, le=1)
    h_splits: Optional[conint(ge=0)] = 0
    w_splits: Optional[conint(ge=0)] = 0
    grayscale: Optional[bool] = False
    kernel_initializer: Optional[str] = "he_normal"
    batch_size: Optional[PositiveInt] = 32
    epochs: Optional[PositiveInt] = 2
    linknet: Optional[bool] = False
    plus_plus: Optional[bool] = False
    deep_supervision: Optional[bool] = False

    @validator("type")
    def check_model_type(cls, v: str):
        if v in implemented_models:
            return v
        raise NotImplementedError(f"The model type must be one of: {', '.join(implemented_models)}")


class SemanticSegmentationInput(BaseModel):
    data: SemanticSegmentationData
    models: List[SemanticSegmentationModelSpec]
