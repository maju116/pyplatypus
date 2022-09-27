"""This module provides us with the tools for preparing the ready-to-use loss functions taken from
pyplatypus.segmentation.loss_functions.segmentation_loss class. These tools are mainly used in the platypus_engine.

Functions
---------
prepare_loss_and_metrics(self, loss: str, metrics: list, n_class: int, background_index: Optional[int] = None)
        Returns the losses and metrics as the functions.

prepare_metrics(self, metrics: list, n_class: int, background_index: Optional[int] = None)
    Returns the metrics as the functions.

prepare_loss_function(loss: str, n_class: int, background_index: Optional[int] = None)
    Returns the ready-to-use loss function in the format expected by the Tensorflow.
"""

from typing import Optional, Callable
from pyplatypus.utils.toolbox import convert_to_snake_case
from pyplatypus.segmentation.loss_functions import segmentation_loss


def prepare_loss_and_metrics(
    loss: str, metrics: list, n_class: int, background_index: Optional[int] = None
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


def prepare_loss_function(loss: str, n_class: int, background_index: Optional[int] = None) -> Callable:
    """
    Returns the ready-to-use loss function in the format expected by the Tensorflow. The function is
    extracted as the attribute of the segmentation_loss function.

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
    loss = convert_to_snake_case(any_case=loss)
    loss_function = getattr(
        segmentation_loss(n_class, background_index), loss
        )
    return loss_function
