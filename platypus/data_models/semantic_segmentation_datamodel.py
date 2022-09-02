from pydantic import BaseModel, validator
from pydantic import PositiveInt, conint, conlist, confloat
from typing import List, Optional, Union, Tuple
from pathlib import Path


from platypus.utils.toolbox import convert_to_snake_case
from platypus.config.input_config import (
    implemented_modes, implemented_losses, implemented_metrics,
    implemented_optimizers, available_activations
    )


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
    metrics: Optional[List[str]] = ["IoU Coefficient"]
    optimizer: Optional[str] = "adam"

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

    @validator("optimizer")
    def check_the_opimizer(cls, v: str):
        v_converted = v.lower()
        if v_converted in implemented_optimizers:
            return v
        raise ValueError(f" The chosen optimizer: {v} is not among the ones available in the Tensorflow!")

class SemanticSegmentationModelSpec(BaseModel):
    name: str
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
    resunet: Optional[bool] = False
    linknet: Optional[bool] = False
    plus_plus: Optional[bool] = False
    deep_supervision: Optional[bool] = False
    use_separable_conv2d: Optional[bool] = True
    use_spatial_dropout2d: Optional[bool] = True
    use_up_sampling2d: Optional[bool] = False
    u_net_conv_block_width: Optional[int] = 2
    activation_layer: Optional[str] = "relu"

    @validator("activation_layer")
    def check_activation_type(cls, v: str):
        if v in available_activations:
            return v
        raise ValueError(f"""
            The selected activation function: {v} is not available in keras! As a note, the activation
            functions' names should be lowercase, maybe that solves the problem?
            """)


class SemanticSegmentationInput(BaseModel):
    data: SemanticSegmentationData
    models: List[SemanticSegmentationModelSpec]
