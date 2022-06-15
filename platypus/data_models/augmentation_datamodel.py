from pydantic import BaseModel, PositiveFloat, conlist, confloat
from typing import Optional


class AugmentationSpecFull(BaseModel):
    ToFloat: ToFloatSpec
    RandomRotate90: RandomRotate90Spec
    Blur: BlurSpec
    GaussianBlur = GaussianBlurSpec
    MedianBlur = MedianBlurSpec
    MotionBlur = MotionBlurSpec
    ErrorAug = ErrorAugSpec
    CLAHE = CLAHESpec
    ChannelDropout = ChannelDropoutSpec
    ChannelShuffle = ChannelShuffleSpec
    

class ToFloatSpec(BaseModel):
    max_value: Optional[PositiveFloat] = 255
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 1.


class RandomRotate90Spec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class BlurSpec(BaseModel):
    blur_limit: Optional[float] = 7
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class GaussianBlurSpec(BaseModel):
    blur_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [3, 7]
    sigma_limit: Optional[float] = 0
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class MedianBlurSpec(BaseModel):
    blur_limit: Optional[PositiveFloat] = 9
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class MotionBlurSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class ErrorAugSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class CLAHESpec(BaseModel):
    clip_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    title_grid_size: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class ChannelDropoutSpec(BaseModel):
    channel_drop_range: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 1]
    fill_value: Optional[float] = 0
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class ChannelShuffle(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class ErrorAugSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5

class ColorJitterSpec(BaseModel):
    brightness: Optional[float] = .2
    contrast: Optional[float] = .2
    saturation: Optional[float] = .2
    hue: Optional[float] = .2
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class DownscaleSpec(BaseModel):
    scale_min: Optional[float] = .25
    scale_max: Optional[float] = .25
    interpolation: Optional[float] = 0
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class EmbossSpec(BaseModel):
    clip_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    title_grid_size: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class EqualizeSpec(BaseModel):
    mode: Optional[str] = "cv"
    by_channel: Optional[bool] = True
    mask: Optional = None  # TODO What type?
    mask_params: Optional[list] = [] # TODO what type?
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class FancyPCASpec(BaseModel):
    alpha: Optional[float] = .1
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class GaussNoiseSpec(BaseModel):
    var_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [10.0, 50.0]
    mean: Optional[float] = 0
    per_channel: Optional[bool] = True
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class HueSaturationValueSpec(BaseModel):
    hue_shift_limit: Optional[PositiveFloat] = 20
    sat_shift_limit: Optional[PositiveFloat] = 30
    val_shift_limit: Optional[PositiveFloat] = 20
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class ISONoiseSpec(BaseModel):
    clip_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    title_grid_size: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [1, 4]
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class InvertImgSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class MultiplicativeNoiseSpec(BaseModel):
    multiplier: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [.9, 1.1]
    per_channel: Optional[bool] = False
    elementwise: Optional[bool] = False
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class NormalizeSpec(BaseModel):
    mean: Optional[conlist(PositiveFloat, min_items=3, max_items=3)] = [0.485, 0.456, 0.406]
    std: Optional[conlist(PositiveFloat, min_items=3, max_items=3)] = [0.229, 0.224, 0.225]
    max_pixel_value: Optional[float] = 255
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = 1.


class RGBShiftSpec(BaseModel):
    r_shift_limit: Optional[PositiveFloat] = 20
    g_shift_limit: Optional[PositiveFloat] = 20
    b_shift_limit: Optional[PositiveFloat] = 20
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class RandomBrightnessContrastSpec(BaseModel):
    brightness_limit: Optional[PositiveFloat] = 0.2
    contrast_limit: Optional[PositiveFloat] = 0.2
    brightness_by_max: Optional[bool] = True
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class RandomFogSpec(BaseModel):
    fog_coef_lower: Optional[PositiveFloat] = 0.3
    fog_coef_upper: Optional[PositiveFloat] = 1
    alpha_coef: Optional[PositiveFloat] = 0.08
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class RandomGammaSpec(BaseModel):
    gamma_limit: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [80, 120]
    eps: Optional[float] = None
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


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


class RandomSnowSpec(BaseModel):
    snow_point_lower: Optional[PositiveFloat] = 0.1
    snow_point_upper: Optional[PositiveFloat] = 0.3
    brightness_coeff: Optional[PositiveFloat] = 2.5
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class RandomShadowSpec(BaseModel):
    shadow_roi: Optional[conlist(PositiveFloat, min_items=4, max_items=4)] = [0, 0.5, 1, 1]
    num_shadows_lower: Optional[PositiveFloat] = 1
    num_shadows_upper: Optional[PositiveFloat] = 2
    shadow_dimension: Optional[float] = 5
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


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


class RandomToneCurveSpec(BaseModel):
    scale: Optional[float] = 0.1
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class AugmentationSpecFull(BaseModel):
    ToFloat: ToFloatSpec
    RandomRotate90: RandomRotate90Spec
    Blur: BlurSpec
    GaussianBlur = GaussianBlurSpec
    MedianBlur = MedianBlurSpec
    MotionBlur = MotionBlurSpec
    ErrorAug = ErrorAugSpec
    CLAHE = CLAHESpec
    ChannelDropout = ChannelDropoutSpec
    ChannelShuffle = ChannelShuffleSpec
    ColorJitter = ColorJitterSpec
    Downscale = DownscaleSpec
    Emboss = EmbossSpec
    Equalize = EqualizeSpec
    FancyPCA = FancyPCASpec
    GaussNoise = GaussNoiseSpec
    HueSaturationValue = HueSaturationValueSpec
    ISONoise = ISONoiseSpec
    InvertImg = InvertImgSpec
    MultiplicativeNoise = MultiplicativeNoiseSpec
    Normalize = NormalizeSpec
    RGBShift = RGBShiftSpec
    RandomBrightnessContrast = RandomBrightnessContrastSpec
    RandomFog = RandomFogSpec
    RandomGamma = RandomGammaSpec
    RandomRain = RandomRainSpec
    RandomSnow = RandomSnowSpec
    RandomShadow = RandomShadowSpec
    RandomSunFlare = RandomSunFlareSpec
    RandomToneCurve = RandomToneCurveSpec
    Sharpen = SharpenSpec
    Solarize = SolarizeSpec

class SharpenSpec(BaseModel):
    alpha: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [0.2, 0.5]
    lightness: Optional[conlist(PositiveFloat, min_items=2, max_items=2)] = [0.5, 1.0]
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class SolarizeSpec(BaseModel):
    threshold: Optional[float] = 128
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5

class SuperpixelsSpec(BaseModel):
    p_replace: Optional[float] = 0.1
    n_segments: Optional[float] = 100
    max_size: Optional[float] = 128
    interpolation: Optional[float] = 1
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


class ToSepiaSpec(BaseModel):
    always_apply: Optional[bool] = False
    p: Optional[confloat(ge=0, le=1)] = .5


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


# =====================================
augmentation:

    CenterCrop:
        height: 30
        width: 30
        always_apply: False
        p: 0.5
    CoarseDropout:
        max_holes: 8
        max_height: 8
        max_width: 8
        min_holes: null
        min_height: null
        min_width: null
        fill_value: 0
        mask_fill_value: None
        always_apply: False
        p: 0.5
    Crop:
        x_min: 0
        y_min: 0
        x_max: 1024
        y_max: 1024
        always_apply: False
        p: 0.5
    CropAndPad:
        px: null
        percent: [30, 50]
        pad_mode: 0
        pad_cval: 0
        pad_cval_mask: 0
        keep_size: True
        sample_independently: True
        interpolation: 1
        always_apply: False
        p: 1.0
    CropNonEmptyMaskIfExists:
        height: 30
        width: 30
        ignore_values: null
        ignore_channels: null
        always_apply: False
        p: 1.0
    ElasticTransform:
        alpha: 1
        sigma: 50
        alpha_affine: 50
        interpolation: 1
        border_mode: 4
        value: null
        mask_value: null
        always_apply: False
        approximate: False
        same_dxdy: False
        p: 0.5
    Flip:
        always_apply: False
        p: 0.5
    GridDistortion:
        num_steps: 5
        distort_limit: 0.3
        interpolation: 1
        border_mode: 4
        value: null
        mask_value: null
        always_apply: False
        p: 0.5
    GridDropout:
        ratio: 0.5
        unit_size_min: null
        unit_size_max: null
        holes_number_x: null
        holes_number_y: null
        shift_x: 0
        shift_y: 0
        random_offset: False
        fill_value: 0
        mask_fill_value: null
        always_apply: False
        p: 0.5
    HorizontalFlip:
        p: 0.5
    MaskDropout:
        max_objects: 1
        image_fill_value: 0
        mask_fill_value: 0
        always_apply: False
        p: 0.5
    OpticalDistortion:
        distort_limit: 0.05
        shift_limit: 0.05
        interpolation: 1
        border_mode: 4
        value: null
        mask_value: null
        always_apply: False
        p: 0.5
    Perspective:
        scale: [0.05, 0.1]
        keep_size: True
        pad_mode: 0
        pad_val: 0
        mask_pad_val: 0
        fit_output: False
        interpolation: 1
        always_apply: False
        p: 0.5
    PiecewiseAffine:
        scale: [0.03, 0.05]
        nb_rows: 4
        nb_cols: 4
        interpolation: 1
        mask_interpolation: 0
        cval: 0
        cval_mask: 0
        mode: 'constant'
        absolute_scale: False
        always_apply: False
        keypoints_threshold: 0.01
        p: 0.5
    RandomCrop:
        height: 30
        width: 30
        always_apply: False
        p: 0.5
    RandomCropNearBBox:
        max_part_shift: [0.3, 0.3]
        cropping_box_key: 'cropping_bbox'
        always_apply: False
        p: 1.0
    RandomGridShuffle:
        grid: [3, 3]
        always_apply: False
        p: 0.5
    RandomResizedCrop:
        height: 30
        width: 30
        scale: [0.08, 1.0]
        ratio: [0.75, 1.3333333333333333]
        interpolation: 1
        always_apply: False
        p: 1.0
    RandomRotate90:
        always_apply: False
        p: 0.5
    RandomSizedBBoxSafeCrop:
        height: 30
        width: 30
        erosion_rate: 0.0
        interpolation: 1
        always_apply: False
        p: 1.0
    Rotate:
        limit: 90
        interpolation: 1
        border_mode: 4
        value: null
        mask_value: null
        always_apply: False
        p: 0.5
    SafeRotate:
        limit: 90
        interpolation: 1
        border_mode: 4
        value: null
        mask_value: null
        always_apply: False
        p: 0.5
    ShiftScaleRotate:
        shift_limit: 0.0625
        scale_limit: 0.1
        rotate_limit: 45
        interpolation: 1
        border_mode: 4
        value: null
        mask_value: null
        shift_limit_x: null
        shift_limit_y: null
        always_apply: False
        p: 0.5
    Transpose:
        always_apply: False
        p: 0.5
    VerticalFlip:
        always_apply: False
        p: 0.5
    FromFloat:
        dtype: 'uint16'
        max_value: 255
        always_apply: False
        p: 1.0
    ToFloat:
        max_value: 255
        always_apply: False
        p: 1.0




