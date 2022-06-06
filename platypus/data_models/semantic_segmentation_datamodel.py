from pydantic import BaseModel, validator
from pydantic import PositiveInt, conint, conlist, contuple, confloat
from typing import List, Optional, Union

from platypus.config.input_config import implemented_models, implemented_modes


class SemanticSegmentationData(BaseModel):
    train_path: str
    validation_path: str
    test_path: str
    colormap: Union[
        List[conlist(conint(ge=0, le=255), min_items=3, max_items=3)],
        List[contuple(conint(ge=0, le=255), min_items=3, max_items=3)]
        ]
    mode: str
    shuffle: bool
    subdirs: conlist(str, min_items=2, max_items=2)
    column_sep: str

    @validator('train_path')
    def check_if_train_path_exists(cls, v: str):
        if v.exists():
            return v
        raise NotADirectoryError("Specified train path does not exist!")

    @validator('validation_path')
    def check_if_validation_path_exists(cls, v: str):
        if v.exists():
            return v
        raise NotADirectoryError("Specified validation path does not exist!")

    @validator('test_path')
    def check_if_test_path_exists(cls, v: str):
        if v.exists():
            return v
        raise NotADirectoryError("Specified test path does not exist!")

    @validator('mode')
    def check_mode_value(cls, v: str):
        if v in implemented_modes:
            return v
        raise ValueError(f"Mode is supposed to be one of: {', '.join(implemented_modes)}!")


class SemanticSegmentationModelSpec(BaseModel):
    name: str
    type: str
    net_h: PositiveInt
    net_w: PositiveInt
    blocks: PositiveInt
    n_class: conint(ge=2)
    filters: PositiveInt
    dropout: confloat(ge=0, le=1)
    grayscale: Optional[bool] = False
    kernel_initializer: Optional[str] = "he_normal"
    batch_size: Optional[PositiveInt] = 32
    epochs: Optional[PositiveInt] = 2

    @validator("type")
    def check_model_type(cls, v: str):
        if v in implemented_models:
            return v
        raise NotImplementedError(f"The model type must be one of: {', '.join(implemented_models)}")


class SemanticSegmentationInput(BaseModel):
    data: SemanticSegmentationData
    models: List[SemanticSegmentationModelSpec]
