"""This script provides pydantic models for both the u-shaped architectures and the data that is to be fed to a model."""

import pydantic
from pydantic import BaseModel, validator
from pydantic import PositiveInt, conint, conlist, confloat
from typing import List, Optional, Union, Tuple, Any
from pathlib import Path

from pyplatypus.utils.toolbox import convert_to_snake_case
from pyplatypus.config.input_config import (
    implemented_modes, implemented_losses, implemented_metrics,
    available_optimizers, available_activations
    )
from pyplatypus.data_models.optimizer_datamodel import AdamSpec


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
    def check_colormap_length(cls, v: list):
        if all([len(c) == 3 for c in v]):
            return v
        raise ValueError("The colormap must consist of three-element lists or tuples!")

    @validator('mode')
    def check_mode_value(cls, v: str):
        if v in implemented_modes:
            return v
        raise ValueError(f"Mode is supposed to be one of: {', '.join(implemented_modes)}!")


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
    loss: Optional[str] = "Iou loss"
    metrics: Optional[List[str]] = ["IoU Coefficient"]
    optimizer: Any = AdamSpec()

    @validator("activation_layer")
    def check_activation_type(cls, v: str):
        if v in available_activations:
            return v
        raise ValueError(f"""
            The selected activation function: {v} is not available in keras! As a note, the activation
            functions' names should be lowercase, maybe that solves the problem?
            """)

    @validator("loss")
    def check_the_loss_name(cls, v: str):
        if convert_to_snake_case(v) in implemented_losses:
            return v
        raise ValueError(f"The chosen loss: {v} is not one of the implemented losses!")

    @validator("optimizer")
    def check_optimizer(cls, v: Any):
        optimizer_name = v.name
        if optimizer_name in available_optimizers:
            return v
        raise ValueError(f" The chosen optimizer: {v} is not among the ones available in the Tensorflow!")

    @validator("metrics")
    def check_the_metrics(cls, v: list):
        v_converted = [convert_to_snake_case(case) for case in v]
        if set(v_converted).issubset(set(implemented_metrics)):
            return v
        raise ValueError(f"The chosen metrics: {', '.join(v)} are not the subset of the implemented ones!")


class SemanticSegmentationInput(BaseModel):
    data: SemanticSegmentationData
    models: List[SemanticSegmentationModelSpec]

    @pydantic.validator("models")
    def check_models_names(cls, v):
        if v:
            model_names = [model.name for model in v]
            unique_names = set(model_names)
            if len(unique_names) == len(model_names):
                return v
            raise ValueError("Models' names have to be unique!")
