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


def concatenate_binary_masks(binary_mask: np.ndarray, colormap: List[Tuple[int, int, int]]):
    """
    Concatenates the binary masks back to the multi-class

    Parameters
    ----------
    mask: np.ndarray
        Segmentation mask.
    colormap: List[Tuple[int, int, int]]
        Class color map.
    """
    n_class = len(colormap)
    for c, i in zip(colormap, range(n_class)):
        class_slice_binary = binary_mask[:, :, i]
        class_slice_mask = np.where(class_slice_binary == 1, c[0], class_slice_binary)
        # Put decoded slice back into the input
        binary_mask[:, :, i] = class_slice_mask
    masks_multiclass = np.split(binary_mask, axis=2, indices_or_sections=n_class)
    return masks_multiclass


def sum_multiclass_masks(masks_multiclass: List[tf.Tensor], colormap: List[Tuple[int, int, int]]):
    n_layers = len(colormap[0])
    masks_multiclass = sum(masks_multiclass)
    # Mimic the original number of layers
    masks_multiclass = np.repeat(masks_multiclass, n_layers, axis=2)
    return masks_multiclass


def transform_probabilities_into_binaries(prediction: np.ndarray):
    max_prob = prediction.max()
    if max_prob > 0:
        prediction_binary = np.where(prediction == max_prob, 1, 0)
    else:
        prediction_binary = prediction
    return prediction_binary
