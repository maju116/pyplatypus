import numpy as np
import tensorflow as tf
from typing import List, Tuple


def convert_to_snake_case(any_case: str):
    """
    Converts the name to the snake case.

    Parameters
    ----------
    any_case: str
        Name to convert e.g. "Dice loss"

    Returns
    -------
    snake_case: str
        Snake case string e.g. "dice_loss"
    """
    snake_case = "_".join(any_case.lower().split(" "))
    return snake_case


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


def transform_probabilities_into_binaries(prediction: np.ndarray):
    prediction_binary = np.apply_along_axis(binary_based_on_arg_max, 2, prediction)
    return prediction_binary


def binary_based_on_arg_max(array: np.array):
    highest_prob = array.max()
    array_binary = np.where(array == highest_prob, 1, 0)
    return array_binary
