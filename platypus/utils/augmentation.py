import numpy as np
from typing import List
import albumentations as A
from albumentations import Blur, GaussianBlur, GlassBlur, MedianBlur, MotionBlur, \
    CLAHE, ChannelDropout, ChannelShuffle, ColorJitter, Downscale, Emboss, Equalize, \
    FancyPCA, GaussNoise, HueSaturationValue, ISONoise, InvertImg, MultiplicativeNoise, \
    Normalize, RGBShift, RandomBrightnessContrast, RandomFog, RandomGamma, RandomRain, \
    RandomSnow, RandomShadow, RandomSunFlare, RandomToneCurve, Sharpen, Solarize, Superpixels, \
    ToSepia

available_methods = ['Blur', 'GaussianBlur', 'GlassBlur', 'MedianBlur', 'MotionBlur',
                     'CLAHE', 'ChannelDropout', 'ChannelShuffle', 'ColorJitter', 'Downscale',
                     'Emboss', 'Equalize', 'FancyPCA', 'GaussNoise', 'HueSaturationValue',
                     'ISONoise', 'InvertImg', 'MultiplicativeNoise', 'Normalize', 'RGBShift',
                     'RandomBrightnessContrast', 'RandomFog', 'RandomGamma', 'RandomRain',
                     'RandomSnow', 'RandomShadow', 'RandomSunFlare', 'RandomToneCurve',
                     'Sharpen', 'Solarize', 'Superpixels', 'ToSepia']


def filter_out_incorrect_methods(
        methods: List[str]
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
    pipes = [globals()[method](**augmentation_dict[method]) for method in correct_methods]
    pipeline = A.Compose(pipes)
    return pipeline
