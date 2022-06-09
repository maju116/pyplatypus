import numpy as np
from typing import List
from collections.abc import KeysView
import albumentations as A

train_available_methods = ['Blur', 'GaussianBlur', 'GlassBlur', 'MedianBlur', 'MotionBlur',
                           'CLAHE', 'ChannelDropout', 'ChannelShuffle', 'ColorJitter', 'Downscale',
                           'Emboss', 'Equalize', 'FancyPCA', 'GaussNoise', 'HueSaturationValue',
                           'ISONoise', 'InvertImg', 'MultiplicativeNoise', 'Normalize', 'RGBShift',
                           'RandomBrightnessContrast', 'RandomFog', 'RandomGamma', 'RandomRain',
                           'RandomSnow', 'RandomShadow', 'RandomSunFlare', 'RandomToneCurve',
                           'Sharpen', 'Solarize', 'Superpixels', 'ToSepia', 'Affine', 'CenterCrop',
                           'CoarseDropout', 'Crop', 'CropAndPad', 'CropNonEmptyMaskIfExists',
                           'ElasticTransform', 'Flip', 'GridDistortion', 'GridDropout', 'HorizontalFlip',
                           'MaskDropout', 'OpticalDistortion', 'Perspective', 'PiecewiseAffine', 'RandomCrop',
                           'RandomCropNearBBox', 'RandomGridShuffle', 'RandomResizedCrop', 'RandomRotate90',
                           'RandomSizedBBoxSafeCrop', 'Rotate', 'SafeRotate', 'ShiftScaleRotate', 'Transpose',
                           'VerticalFlip', 'FromFloat', 'ToFloat']
validation_test_available_methods = ['FromFloat', 'ToFloat', 'InvertImg']


def filter_out_incorrect_methods(
        methods: KeysView,
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
    return [m for m in methods if m in set(available_methods)]


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
    correct_methods = filter_out_incorrect_methods(augmentation_dict.keys(), train)
    augmentation_dict = {your_key: augmentation_dict[your_key] for your_key in correct_methods}
    pipes = [getattr(A, method)(**augmentation_dict[method]) for method in correct_methods]
    pipeline = A.Compose(pipes, p=1.0)
    return pipeline
