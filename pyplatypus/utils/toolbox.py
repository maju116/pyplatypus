"""This module offers the set of tools shared between the various tasks connected by the fact of being related to the Computer Vision topic.

Functions
---------
convert_to_snake_case(any_case: str)
    Converts any given string to the snake case.

split_masks_into_binary(mask: np.ndarray, colormap: List[Tuple[int, int, int]])
    Splits multi-class mask into binary masks.

concatenate_binary_masks(binary_mask: np.ndarray, colormap: List[Tuple[int, int, int]])
    Concatenates the binary masks back to the multi-class.

transform_probabilities_into_binaries(prediction: np.ndarray)
    Over the last dimension, the function sets the argmax of the probabilities,
    sets the value for it to be one whilst zeroing out the rest of elements.

binary_based_on_arg_max(array: np.array)
    Replaces the highest probability with one and zeros the rest.
"""

import numpy as np
from typing import List, Tuple


def convert_to_snake_case(any_case: str) -> str:
    """
    Converts any given string to the snake case.

    Parameters
    ----------
    any_case: str
        String to convert.

    Returns
    -------
    snake_case: str
        Snake case string.

    Examples
    --------
    >>> convert_to_snake_case("Dice loss")
    'dice_loss'
    """
    snake_case = "_".join(any_case.lower().split(" "))
    return snake_case


def convert_to_camel_case(any_case: str) -> str:
    """
    Converts any given string to the snake case.

    Parameters
    ----------
    any_case: str
        String to convert.

    Returns
    -------
    camel_case: str
        Camel case string.

    Examples
    --------
    >>> convert_to_snake_case("Dice loss")
    'DiceLoss'
    """
    if "_" in any_case:
        splitted = any_case.split("_")
    elif " " in any_case:
        splitted = any_case.split(" ")
    else:
        raise ValueError(f"Unable to convert: {any_case} to CamelCase! No underscore or space detected!")
    splitted = [s.title() for s in splitted]
    camel_case = "".join(splitted)
    return camel_case


def split_masks_into_binary(
        mask: np.ndarray,
        colormap: List[Tuple[int, int, int]]
        ) -> np.ndarray:
    """
    Splits multi-class mask into binary masks.

    Parameters
    ----------
    mask: np.ndarray
        Segmentation mask.
    colormap: List[Tuple[int, int, int]]
        Class color map.
    """
    return np.stack([np.all(mask == c, axis=-1) * 1 for c in colormap], axis=-1)


def concatenate_binary_masks(
        binary_mask: np.ndarray,
        colormap: List[Tuple[int, int, int]]
        ) -> np.ndarray:
    """
    Concatenates the binary masks back to the multi-class

    Parameters
    ----------
    binary_mask: np.ndarray
        Binary segmentation mask.
    colormap: List[Tuple[int, int, int]]
        Class color map.
    """
    multiclass_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3))
    for count, color in enumerate(colormap):
        idx = np.where(binary_mask[:, :, count] == 1)
        multiclass_mask[idx] = color
    return multiclass_mask


def transform_probabilities_into_binaries(prediction: np.ndarray) -> np.array:
    """Over the last dimension, the function sets the argmax of the probabilities,
    sets the value for it to be one whilst zeroing out the rest of elements.

    Parameters
    ----------
    prediction : np.ndarray
        Multidimensional array containing the probabilities of each pixel belonging to each class.

    Returns
    -------
    prediction_binary: np.array
        Array containing only binary values.

    Examples
    --------
    >>> transform_probabilities_into_binaries(np.array([0.9, 0.8, 0.7]).reshape((1, 1, 3)))
    array([[[1, 0, 0]]])
    """
    prediction_binary = np.apply_along_axis(binary_based_on_arg_max, 2, prediction)
    return prediction_binary


def binary_based_on_arg_max(array: np.array):
    """Replaces the highest probability with one and zeros the rest.

    Parameters
    ----------
    array : np.array
        Input array containing the probablities between zero and one.

    Returns
    -------
    array_binary: np.array
        Binary array.

    Examples
    --------
    >>> binary_based_on_arg_max(np.array([0.9, 0.8, 0.7]))
    array([1, 0, 0])
    """
    highest_prob = array.max()
    array_binary = np.where(array == highest_prob, 1, 0)
    return array_binary
