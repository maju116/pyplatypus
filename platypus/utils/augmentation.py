import numpy as np
from typing import List
from collections.abc import KeysView
import albumentations as A

from platypus.config.augmentation_config import train_available_methods, validation_test_available_methods


def filter_out_incorrect_methods(
        augmentation_dict: dict,
        train: bool
) -> List[str]:
    """
    Filters out incorrect names of augmentation methods to be used.

    Args:
        methods (List[str]): List of names with augmentation methods to be used.
        train (bool): Should the train methods list be used.

    Returns:
        List of correct names with augmentation methods.
    """
    if train:
        available_methods = train_available_methods
    else:
        available_methods = validation_test_available_methods

    methods = augmentation_dict.keys()
    chosen_transformations = [m for m in augmentation_dict.keys() if augmentation_dict.get(m) is not None]

    return [m for m in methods if (m in available_methods) and (m in chosen_transformations)]


def create_augmentation_pipeline(
        augmentation_dict: dict,
        train: bool,
) -> A.core.composition.Compose:
    """
    Create augmentation pipeline based on dictionary.

    Args:
        augmentation_dict (dict): Augmentation dictionary.
        train (bool): Should the train methods list be used.

    Returns:
        Augmentation pipeline
    """
    correct_methods = filter_out_incorrect_methods(augmentation_dict, train)
    augmentation_dict = {your_key: augmentation_dict[your_key] for your_key in correct_methods}
    pipes = [getattr(A, method)(**dict(augmentation_dict[method])) for method in correct_methods]
    pipeline = A.Compose(pipes, p=1.0)
    return pipeline
