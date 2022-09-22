"""This script defines the available augmentation methods, camel case names are compliant with the methods available
in the Albumentations package."""

train_available_methods = [
    'Blur', 'GaussianBlur', 'GlassBlur', 'MedianBlur', 'MotionBlur',
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
    'VerticalFlip', 'FromFloat', 'ToFloat'
    ]
validation_test_available_methods = ['FromFloat', 'ToFloat', 'InvertImg']
