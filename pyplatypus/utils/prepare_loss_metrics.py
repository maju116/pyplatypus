"""This module provides us with the tools for preparing the ready-to-use loss functions taken from
pyplatypus.segmentation.loss_functions.SegmentationLoss class. These tools are mainly used in the PlatypusEngine.

Functions
---------
prepare_loss_and_metrics(self, loss: str, metrics: list, n_class: int, background_index: Optional[int] = None)
        Returns the losses and metrics as the functions.

prepare_metrics(self, metrics: list, n_class: int, background_index: Optional[int] = None)
    Returns the metrics as the functions.

prepare_loss_function(loss: str, n_class: int, background_index: Optional[int] = None)
    Returns the ready-to-use loss function in the format expected by the Tensorflow.
"""

import pydantic
from typing import Optional, Callable, Union
from pyplatypus.utils.toolbox import convert_to_camel_case
import pyplatypus.segmentation.extended_loss_functions as ELF
from pyplatypus.data_models.optimizer_datamodel import (
    AdadeltaSpec, AdagradSpec, AdamSpec, AdamaxSpec, FtrlSpec, NadamSpec, RMSpropSpec, SGDSpec
    )
from pyplatypus.data_models.callbacks_datamodel import (
    EarlyStoppingSpec, ModelCheckpointSpec, ReduceLROnPlateauSpec, TensorBoardSpec, BackupAndRestoreSpec,
    TerminateOnNaNSpec, CSVLoggerSpec, ProgbarLoggerSpec
)

import tensorflow.keras.optimizers as TOPT
import pyplatypus.utils.extended_tensorflow_callbacks as TCB
from tensorflow.keras.callbacks import Callback


def prepare_loss_and_metrics(
    loss: pydantic.BaseModel, metrics: list, n_class: int, background_index: Optional[int] = None
        ) -> tuple:
    """Returns the losses and metrics as the functions.

    Parameters
    ----------
    loss: str
        Name of any format.
    metrics: list
        Names (any case) of the metrics to be used in the training and validation performance assessment.
    n_class: int
        Indicates to how many classes each pixel could be classified i.e. the number of possible classes.
    background_index: int
        Used mainly by the remove_index method points to the layer storing the background probabilities.

    Returns
    -------
    training_loss: function
        Ready-to-use function calculating the loss given the classes' probabilities and the ground truth.
    metrics_to_apply: list
        List of functions serving as the trainining/validation sets evaluators.
    """
    training_loss = prepare_loss_function(
        loss=loss, n_class=n_class, background_index=background_index
        )
    metrics_to_apply = prepare_metrics(metrics, n_class, background_index)
    return training_loss, metrics_to_apply


def prepare_metrics(
    metrics: list, n_class: int, background_index: Optional[int] = None
        ) -> list:
    """
    Returns the metrics as the functions.

    Parameters
    ----------
    metrics: list
        Names (any case) of the metrics to be used in the training and validation performance assessment.
    n_class: int
        Indicates to how many classes each pixel could be classified i.e. the number of possible classes.
    background_index: int
        Used mainly by the remove_index method points to the layer storing the background probabilities.

    Returns
    -------
    metrics_to_apply: list
        List of functions serving as the trainining/validation sets evaluators.
    """
    metrics_to_apply = ["categorical_crossentropy"]
    for metric in metrics:
        metric_function = prepare_loss_function(
            loss=metric, n_class=n_class, background_index=background_index
            )
        metrics_to_apply.append(metric_function)
    metrics_to_apply = list(set(metrics_to_apply))
    return metrics_to_apply


def prepare_loss_function(loss: pydantic.BaseModel, n_class: int, background_index: Optional[int] = None) -> Callable:
    """
    Returns the ready-to-use loss function in the format expected by the Tensorflow. The function is
    extracted as the attribute of the SegmentationLoss function.

    Parameters
    ----------
    loss: str
        Name of any format.
    n_class: int
        Indicates to how many classes each pixel could be classified i.e. the number of possible classes.
    background_index: int
        Used mainly by the remove_index method points to the layer storing the background probabilities.

    Returns
    -------
    loss_function: function
        Ready-to-use function calculating the loss given the classes' probabilities and the ground truth.
    """
    cc_loss_name = convert_to_camel_case(loss.name)
    loss_name = loss.name
    getter = getattr(ELF, f"{cc_loss_name}Getter")(n_class, background_index, input_dict=loss.dict())
    loss_function = getattr(getter, loss_name)
    return loss_function


def prepare_optimizer(
    optimizer: Union[
        AdadeltaSpec, AdagradSpec, AdamSpec, AdamaxSpec, FtrlSpec, NadamSpec, RMSpropSpec, SGDSpec
        ]
        ) -> TOPT.Optimizer:
    """Basing on the optimizer name, function takes the desired class from tensorflow backend and
    provides it with the user-specified or default parameters.

    Parameters
    ----------
    optimizer : Union[ AdadeltaSpec, AdagradSpec, AdamSpec, AdamaxSpec, FtrlSpec, NadamSpec, RMSpropSpec, SGDSpec ]
        Optimizer specification, parsed via pydantic.

    Returns
    -------
    initialized_optimizer: TOPT.Optimizer
        Inheriting from the Optimizer base class.
    """
    template_optimizer = getattr(TOPT, optimizer.name)
    initialized_optimizer = template_optimizer(**optimizer.dict())
    return initialized_optimizer


def prepare_callbacks_list(callbacks_specs: list) -> list:
    """Fills in the list of callbacks, optionally returning empty list, which is Tensorflow-compliant.

    Parameters
    ----------
    callbacks_specs : list
        List containing the Pydatnic-powered specificaions for each of the callbacks contained in the input config.

    Returns
    -------
    callbacks: list
        List containing the Tensorflow's backend native objects, defining the callbacks to be used in the training.
    """
    callbacks = []
    for callback in callbacks_specs:
        callbacks.append(prepare_callback(callback=callback))
    return callbacks


def prepare_callback(
    callback: Union[
        EarlyStoppingSpec, ModelCheckpointSpec, ReduceLROnPlateauSpec, TensorBoardSpec, BackupAndRestoreSpec,
        TerminateOnNaNSpec, CSVLoggerSpec, ProgbarLoggerSpec
        ]
        ) -> Callback:
    """Basing on the callback name, function takes the desired class from tensorflow backend and
    provides it with the user-specified or default parameters.

    Parameters
    ----------
    callback : Union[
        EarlyStoppingSpec, ModelCheckpointSpec, ReduceLROnPlateauSpec, TensorBoardSpec, BackupAndRestoreSpec,
        TerminateOnNaNSpec, CSVLoggerSpec, ProgbarLoggerSpec
        ]
        Callback specification, parsed via pydantic.

    Returns
    -------
    initialized_optimizer: Callback
        Inheriting from the Callback base class.
    """
    template_callback = getattr(TCB, f"{callback.name}Extension")
    initialized_callback = template_callback(callback.dict())
    return initialized_callback
