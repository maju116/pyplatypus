"""The module delivers tools related strictly to the topic of augmentation.

Functions
---------
filter_out_incorrect_methods(augmentation_dict: dict, train: bool)
    Filters the names of augmentations methods that are yet to be implemented out of the input list.

create_augmentation_pipeline(augmentation_dict: dict, train: bool)
    Creates augmentation pipeline based on dictionary.
"""

from typing import List, Optional, Any
import albumentations as A
from pyplatypus.config.augmentation_config import train_available_methods, validation_test_available_methods


def filter_out_incorrect_methods(augmentation: list, train: bool) -> List[str]:
    """
    Filters the names of augmentations methods that are yet to be implemented out of the input list.

    Parameters
    ----------
    methods: list
        List of names of the augmentation methods to be used.
    train: bool
        Should the train methods list be used, essentialy it switches us between the
        lists of valid methods.

    Returns
    -------
        valid_methods: list
            List produced out of the inpur of correct names of the augmentation methods.
    """
    if train:
        available_methods = train_available_methods
    else:
        available_methods = validation_test_available_methods
    augmentations_valid = [
        augmentation_spec for augmentation_spec in augmentation if augmentation_spec.name in available_methods
        ]
    return augmentations_valid


def create_augmentation_pipeline(augmentation: list, train: bool) -> A.core.composition.Compose:
    """
    Creates augmentation pipeline based on dictionary. It is done by importing the certain classes from the Albumentations
    module. This is why it is crucial to use the proper names here hence the additional validation. The incorrect keys are
    deleted from the dictionary.

    Parameters
    ----------
    augmentation_dict: dict
        Augmentation dictionary, connecting the methods names to theirs configs.
        Refer to platypus/data_models/augmentation_datamodel.py to learn about their exact structure.
    train: bool
        Switches between different lists of allowed methods.

    Returns
    -------
    pipeline: Albumetations.core.composition.Compose class
        The iterator-like object compliant with the data generators present in the PyPlatypus.
    """
    augmentation_list = filter_out_incorrect_methods(augmentation, train)
    pipes = []
    for method in augmentation_list:
        uinitialized_transform = getattr(A, method.name)
        transform_params = method.copy().dict()
        transform_params.pop("name")
        initialized_transform = uinitialized_transform(**transform_params)
        pipes.append(initialized_transform)
    pipeline = A.Compose(pipes, p=1.0)
    return pipeline


def prepare_augmentation_pipelines(config: dict) -> tuple[Optional[Any], Optional[Any]]:
    """
    Prepares the pipelines consisting of the transforms taken from the albumentations module.

    Parameters
    ----------
    config: dict
        Config steering the workflow, it is checked for the presence of the "augmentation" key.

    Returns
    -------
    augmentation_pipelines: Tuple[albumentations.Compose]
        Composed of the augmentation pipelines.
    """
    if config.augmentation is not None:
        train_augmentation_pipeline = create_augmentation_pipeline(
            augmentation=config.augmentation,
            train=True
            )
        validation_augmentation_pipeline = create_augmentation_pipeline(
            augmentation=config.augmentation,
            train=False
            )
    else:
        train_augmentation_pipeline = None
        validation_augmentation_pipeline = None
    pipelines = (train_augmentation_pipeline, validation_augmentation_pipeline)
    return pipelines
