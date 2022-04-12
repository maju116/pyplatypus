import numpy as np
from typing import List
from collections.abc import KeysView
import albumentations as A

available_methods = ['Blur', 'GaussianBlur', 'GlassBlur', 'MedianBlur', 'MotionBlur',
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


def filter_out_incorrect_methods(
        methods: KeysView
) -> List[str]:
    """
    Filters out incorrect names of augmentation methods to be used.

    Args:
        methods (List[str]): List of names with augmentation methods to be used.

    Returns:
        List of correct names with augmentation methods.
    """
    return list(set(methods).intersection(available_methods))


def create_augmentation_pipeline(
        augmentation_dict: dict
) -> A.core.composition.Compose:
    """
    Create augmentation pipeline based on dictionary.

    Args:
        augmentation_dict (dict): Augmentation dictionary.

    Returns:
        Augmentation pipeline
    """
    correct_methods = filter_out_incorrect_methods(augmentation_dict.keys())
    augmentation_dict = {your_key: augmentation_dict[your_key] for your_key in correct_methods}
    pipes = [getattr(A, method)(**augmentation_dict[method]) for method in correct_methods]
    pipeline = A.Compose(pipes, p=1.0)
    return pipeline
