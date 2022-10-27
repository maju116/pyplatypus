"""Here the pydantic models represents configuration of each implemented transformation. There are all later used for
composing the AugmentationSpecFull data model used while building the augmentation pipelines in the platypus engine."""

from pydantic import BaseModel, PositiveFloat, conlist, confloat
from typing import Optional, List


class BlurSpec(BaseModel):
    blur_limit: Optional[float] = 7
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "Blur"


class GaussianBlurSpec(BaseModel):
    blur_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [3, 7]
    sigma_limit: Optional[float] = 0
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "GaussianBlur"


class MedianBlurSpec(BaseModel):
    blur_limit: Optional[PositiveFloat] = 9
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "MedianBlur"


class MotionBlurSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "MotionBlur"


class ErrorAugSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "ErrorAug"


class CLAHESpec(BaseModel):
    clip_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    title_grid_size: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "CLAHE"


class ChannelDropoutSpec(BaseModel):
    channel_drop_range: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 1]
    fill_value: Optional[float] = 0
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "ChannelDropout"


class ChannelShuffleSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "ChannelShuffle"


class ColorJitterSpec(BaseModel):
    brightness: Optional[float] = .2
    contrast: Optional[float] = .2
    saturation: Optional[float] = .2
    hue: Optional[float] = .2
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "ColorJitter"


class DownscaleSpec(BaseModel):
    scale_min: Optional[float] = .25
    scale_max: Optional[float] = .25
    interpolation: Optional[float] = 0
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "Downscale"


class EmbossSpec(BaseModel):
    clip_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    title_grid_size: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "Emboss"


class EqualizeSpec(BaseModel):
    mode: Optional[str] = "cv"
    by_channel: Optional[bool] = True
    mask: Optional[float] = None
    mask_params: Optional[List[str]] = []
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "Equalize"


class FancyPCASpec(BaseModel):
    alpha: Optional[float] = .1
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "FancyPCA"


class GaussNoiseSpec(BaseModel):
    var_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [10.0, 50.0]
    mean: Optional[float] = 0
    per_channel: Optional[bool] = True
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "GaussianNoise"


class HueSaturationValueSpec(BaseModel):
    hue_shift_limit: Optional[PositiveFloat] = 20
    sat_shift_limit: Optional[PositiveFloat] = 30
    val_shift_limit: Optional[PositiveFloat] = 20
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "HueSaturationValue"


class ISONoiseSpec(BaseModel):
    clip_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    title_grid_size: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "ISONoise"


class InvertImgSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "InvertImg"


class MultiplicativeNoiseSpec(BaseModel):
    multiplier: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [.9, 1.1]
    per_channel: Optional[bool] = False
    elementwise: Optional[bool] = False
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "MultiplicativeNoise"


class NormalizeSpec(BaseModel):
    mean: Optional[conlist(PositiveFloat, min_items=3, max_items=3)] = [0.485, 0.456, 0.406]
    std: Optional[conlist(PositiveFloat, min_items=3, max_items=3)] = [0.229, 0.224, 0.225]
    max_pixel_value: Optional[float] = 255
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 1.
    name: str = "Normalize"


class RGBShiftSpec(BaseModel):
    r_shift_limit: Optional[PositiveFloat] = 20
    g_shift_limit: Optional[PositiveFloat] = 20
    b_shift_limit: Optional[PositiveFloat] = 20
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "RGBShift"


class RandomBrightnessContrastSpec(BaseModel):
    brightness_limit: Optional[PositiveFloat] = 0.2
    contrast_limit: Optional[PositiveFloat] = 0.2
    brightness_by_max: Optional[bool] = True
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "RandomBrightnessContrast"


class RandomFogSpec(BaseModel):
    fog_coef_lower: Optional[PositiveFloat] = 0.3
    fog_coef_upper: Optional[PositiveFloat] = 1
    alpha_coef: Optional[PositiveFloat] = 0.08
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "RandomFog"


class RandomGammaSpec(BaseModel):
    gamma_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [80, 120]
    eps: Optional[float] = None
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "RandomGamma"


class RandomRainSpec(BaseModel):
    slant_lower: Optional[float] = -10
    slant_upper: Optional[float] = 10
    drop_length: Optional[float] = 20
    drop_width: Optional[float] = 1
    drop_color: Optional[conlist(PositiveFloat, min_items=3, max_items=3)] = [200, 200, 200]
    blur_value: Optional[float] = 7
    brightness_coefficient: Optional[float] = .7
    rain_type: Optional[float] = None
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "RandomRain"


class RandomSnowSpec(BaseModel):
    snow_point_lower: Optional[PositiveFloat] = 0.1
    snow_point_upper: Optional[PositiveFloat] = 0.3
    brightness_coeff: Optional[PositiveFloat] = 2.5
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "RandomSnow"


class RandomShadowSpec(BaseModel):
    shadow_roi: Optional[conlist(PositiveFloat, min_items=4, max_items=4)] = [0, 0.5, 1, 1]
    num_shadows_lower: Optional[PositiveFloat] = 1
    num_shadows_upper: Optional[PositiveFloat] = 2
    shadow_dimension: Optional[float] = 5
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "RandomShadow"


class RandomSunFlareSpec(BaseModel):
    shadow_roi: Optional[conlist(PositiveFloat, min_items=4, max_items=4)] = [0, 0, 1, 0.5]
    angle_lower: Optional[PositiveFloat] = 0
    angle_upper: Optional[PositiveFloat] = 1
    num_flare_circles_lower: Optional[float] = 6
    num_flare_circles_upper: Optional[float] = 10
    src_radius: Optional[float] = 400
    src_color: Optional[conlist(PositiveFloat, min_items=3, max_items=3)] = [255, 255, 255]
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "RandomSunFlare"


class RandomToneCurveSpec(BaseModel):
    scale: Optional[float] = 0.1
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "RandomToneCurve"


class SharpenSpec(BaseModel):
    alpha: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [0.2, 0.5]
    lightness: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [0.5, 1.0]
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "Sharpen"


class SolarizeSpec(BaseModel):
    threshold: Optional[float] = 128
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "Solarize"


class SuperpixelsSpec(BaseModel):
    p_replace: Optional[float] = 0.1
    n_segments: Optional[float] = 100
    max_size: Optional[float] = 128
    interpolation: Optional[float] = 1
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "Superpixels"


class ToSepiaSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "ToSepia"


class AffineSpec(BaseModel):
    scale: Optional[float] = None
    translate_percent: Optional[float] = None
    translate_px: Optional[float] = None
    rotate: Optional[bool] = None
    shear: Optional[bool] = None
    interpolation: Optional[float] = 1
    mask_interpolation: Optional[float] = 0
    cval: Optional[float] = 0
    cval_mask: Optional[float] = 0
    mode: Optional[float] = 0
    fit_output: Optional[bool] = False
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "Affine"


class CenterCropSpec(BaseModel):
    height: Optional[PositiveFloat] = 30
    width: Optional[PositiveFloat] = 30
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "CenterCrop"


class CoarseDropoutSpec(BaseModel):
    max_holes: Optional[PositiveFloat] = 8
    max_height: Optional[PositiveFloat] = 8
    max_width: Optional[PositiveFloat] = 8
    max_holes: Optional[PositiveFloat] = None
    max_height: Optional[PositiveFloat] = None
    max_width: Optional[PositiveFloat] = None
    fill_value: Optional[float] = 0
    mask_fill_value: Optional[float] = None
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "CoarseDropout"


class CropSpec(BaseModel):
    x_min: Optional[PositiveFloat] = 0
    y_min: Optional[PositiveFloat] = 0
    x_max: Optional[PositiveFloat] = 1024
    y_max: Optional[PositiveFloat] = 1024
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "Crop"


class CropAndPadSpec(BaseModel):
    px: Optional[float] = None
    percent: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [30, 50]
    pad_mode: Optional[float] = 0
    pad_cval: Optional[float] = 1024
    pad_cval_mask: Optional[float] = 0
    keep_size: Optional[bool] = True
    sample_independently: Optional[bool] = True
    interpolation: Optional[PositiveFloat] = 1
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 1
    name: str = "CropAndPad"


class CropNonEmptyMaskIfExistsSpec(BaseModel):
    height: Optional[PositiveFloat] = 30
    width: Optional[PositiveFloat] = 30
    ignore_values: Optional[bool] = None
    ignore_channels: Optional[bool] = None
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 1
    name: str = "CropNonEmptyMaskIfExists"


class ElasticTransformSpec(BaseModel):
    alpha: Optional[float] = 1
    sigma: Optional[float] = 50
    alpha_affine: Optional[float] = 50
    interpolation: Optional[float] = 1
    border_mode: Optional[float] = 4
    value: Optional[float] = None
    mask_value: Optional[float] = None
    approximate: Optional[bool] = False
    same_dxdy: Optional[bool] = False
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "ElasticTransform"


class FlipSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "Flip"


class GridDistortionSpec(BaseModel):
    num_steps: Optional[PositiveFloat] = 5
    distort_limit: Optional[float] = .3
    interpolation: Optional[float] = 1
    border_mode: Optional[float] = 4
    value: Optional[float] = None
    mask_value: Optional[float] = None
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "GridDistortion"


class GridDropoutSpec(BaseModel):
    ratio: Optional[float] = 0.5
    unit_size_min: Optional[PositiveFloat] = None
    unit_size_max: Optional[PositiveFloat] = None
    holes_number_x: Optional[PositiveFloat] = None
    holes_number_y: Optional[PositiveFloat] = None
    shift_x: Optional[float] = 0
    shift_y: Optional[float] = 0
    random_offset: Optional[bool] = False
    fill_value: Optional[float] = 0
    mask_fill_value: Optional[float] = None
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "GridDropout"


class HorizontalFlipSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "HorizontalFlip"


class MaskDropoutSpec(BaseModel):
    max_objects: Optional[PositiveFloat] = 1
    image_fill_value: Optional[float] = 0
    mask_fill_value: Optional[float] = 0
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "MaskDropout"


class OpticalDistortionSpec(BaseModel):
    distort_limit: Optional[float] = 0.05
    shift_limit: Optional[float] = 0.05
    interpolation: Optional[float] = 1
    border_mode: Optional[float] = 4
    value: Optional[float] = None
    mask_value: Optional[float] = None
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "OpticalDistortion"


class PerspectiveSpec(BaseModel):
    scale: Optional[conlist(float, min_items=2, max_items=2)] = [0.05, 0.1]
    keep_size: Optional[bool] = True
    pad_mode: Optional[float] = 0
    pad_val: Optional[float] = 0
    mask_pad_val: Optional[float] = 0
    fit_output: Optional[bool] = False
    interpolation: Optional[float] = 1
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "Perspective"


class PiecewiseAffineSpec(BaseModel):
    scale: Optional[conlist(float, min_items=2, max_items=2)] = [0.03, 0.05]
    nb_rows: Optional[float] = 4
    nb_cols: Optional[float] = 4
    interpolation: Optional[float] = 1
    mask_interpolation: Optional[float] = 0
    cval: Optional[float] = 0
    cval_mask: Optional[float] = 0
    mode: Optional[str] = "constant"
    absolute_scale: Optional[bool] = False
    keypoints_threshold: Optional[float] = 0.01
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "PiecewiseAffine"


class RandomCropSpec(BaseModel):
    height: Optional[PositiveFloat] = 30
    width: Optional[PositiveFloat] = 30
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "RandomCrop"


class RandomCropNearBBoxSpec(BaseModel):
    max_part_shift: Optional[conlist(float, min_items=2, max_items=2)] = [0.3, 0.3]
    cropping_box_key: Optional[str] = "cropping_bbox"
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 1
    name: str = "RandomCropNearBBox"


class RandomGridShuffleSpec(BaseModel):
    grid: Optional[conlist(float, min_items=2, max_items=2)] = [3, 3]
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5
    name: str = "RandomGridShuffle"


class RandomResizedCropSpec(BaseModel):
    height: Optional[PositiveFloat] = 30
    width: Optional[PositiveFloat] = 30
    scale: Optional[conlist(float, min_items=2, max_items=2)] = [0.08, 1.0]
    ratio: Optional[conlist(float, min_items=2, max_items=2)] = [0.75, 1.3333333333333333]
    interpolation: Optional[float] = 1
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 1
    name: str = "RandomResizedCrop"


class RandomRotate90Spec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 1
    name: str = "RandomRotate90"


class RandomSizedBBoxSafeCropSpec(BaseModel):
    height: Optional[PositiveFloat] = 30
    width: Optional[PositiveFloat] = 30
    erosion_rate: Optional[float] = 0.0
    interpolation: Optional[float] = 1
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 1
    name: str = "RandomSizedBBoxSafeCrop"


class RotateSpec(BaseModel):
    limit: Optional[PositiveFloat] = 90
    interpolation: Optional[float] = 1
    border_mode: Optional[float] = 4
    value: Optional[float] = None
    mask_value: Optional[float] = None
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 0.5
    name: str = "Rotate"


class SafeRotateSpec(BaseModel):
    limit: Optional[PositiveFloat] = 90
    interpolation: Optional[float] = 1
    border_mode: Optional[float] = 4
    value: Optional[float] = None
    mask_value: Optional[float] = None
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 0.5
    name: str = "SafeRotate"


class ShiftRotateSpec(BaseModel):
    shift_limit: Optional[PositiveFloat] = 0.0625
    scale_limit: Optional[PositiveFloat] = 0.1
    rotate_limit: Optional[PositiveFloat] = 45
    interpolation: Optional[float] = 1
    border_mode: Optional[float] = 4
    value: Optional[float] = None
    mask_value: Optional[float] = None
    shift_limit_x: Optional[float] = None
    shift_limit_y: Optional[float] = None
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 0.5
    name: str = "ShiftRotate"


class TransposeSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 0.5
    name: str = "Transpose"


class VerticalFlipSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 0.5
    name: str = "VerticalFlip"


class FromFloatSpec(BaseModel):
    dtype: Optional[str] = "uint16"
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 1
    name: str = "FromFloat"


class ToFloatSpec(BaseModel):
    max_value: Optional[float] = 255
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 1
    name: str = "ToFloat"


class AugmentationSpecFull(BaseModel):
    Blur: Optional[BlurSpec] = None
    GaussianBlur: Optional[GaussianBlurSpec] = None
    MedianBlur: Optional[MedianBlurSpec] = None
    MotionBlur: Optional[MotionBlurSpec] = None
    ErrorAug: Optional[ErrorAugSpec] = None
    CLAHE: Optional[CLAHESpec] = None
    ChannelDropout: Optional[ChannelDropoutSpec] = None
    ChannelShuffle: Optional[ChannelShuffleSpec] = None
    ColorJitter: Optional[ColorJitterSpec] = None
    Downscale: Optional[DownscaleSpec] = None
    Emboss: Optional[EmbossSpec] = None
    Equalize: Optional[EqualizeSpec] = None
    FancyPCA: Optional[FancyPCASpec] = None
    GaussNoise: Optional[GaussNoiseSpec] = None
    HueSaturationValue: Optional[HueSaturationValueSpec] = None
    ISONoise: Optional[ISONoiseSpec] = None
    InvertImg: Optional[InvertImgSpec] = None
    MultiplicativeNoise: Optional[MultiplicativeNoiseSpec] = None
    Normalize: Optional[NormalizeSpec] = None
    RGBShift: Optional[RGBShiftSpec] = None
    RandomBrightnessContrast: Optional[RandomBrightnessContrastSpec] = None
    RandomFog: Optional[RandomFogSpec] = None
    RandomGamma: Optional[RandomGammaSpec] = None
    RandomRain: Optional[RandomRainSpec] = None
    RandomSnow: Optional[RandomSnowSpec] = None
    RandomShadow: Optional[RandomShadowSpec] = None
    RandomSunFlare: Optional[RandomSunFlareSpec] = None
    RandomToneCurve: Optional[RandomToneCurveSpec] = None
    Sharpen: Optional[SharpenSpec] = None
    Solarize: Optional[SolarizeSpec] = None
    Superpixels: Optional[SuperpixelsSpec] = None
    Affine: Optional[AffineSpec] = None
    CenterCrop: Optional[CenterCropSpec] = None
    CoarseDropout: Optional[CoarseDropoutSpec] = None
    Crop: Optional[CropSpec] = None
    CropAndPad: Optional[CropAndPadSpec] = None
    CropNonEmptyMaskIfExists: Optional[CropNonEmptyMaskIfExistsSpec] = None
    ElasticTransform: Optional[ElasticTransformSpec] = None
    Flip: Optional[FlipSpec] = None
    GridDistortion: Optional[GridDistortionSpec] = None
    GridDropout: Optional[GridDropoutSpec] = None
    HorizontalFlip: Optional[HorizontalFlipSpec] = None
    MaskDropout: Optional[MaskDropoutSpec] = None
    OpticalDistortion: Optional[OpticalDistortionSpec] = None
    Perspective: Optional[PerspectiveSpec] = None
    PiecewiseAffine: Optional[PiecewiseAffineSpec] = None
    RandomCrop: Optional[RandomCropSpec] = None
    RandomCropNearBBox: Optional[RandomCropNearBBoxSpec] = None
    RandomGridShuffle: Optional[RandomGridShuffleSpec] = None
    RandomResizedCrop: Optional[RandomResizedCropSpec] = None
    RandomRotate90: Optional[RandomRotate90Spec] = None
    RandomSizedBBoxSafeCrop: Optional[RandomSizedBBoxSafeCropSpec] = None
    Rotate: Optional[RotateSpec] = None
    SafeRotate: Optional[SafeRotateSpec] = None
    ShiftRotate: Optional[ShiftRotateSpec] = None
    Transpose: Optional[TransposeSpec] = None
    VerticalFlip: Optional[VerticalFlipSpec] = None
    FromFloat: Optional[FromFloatSpec] = None
    ToFloat: Optional[ToFloatSpec] = None
