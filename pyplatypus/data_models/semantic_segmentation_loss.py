from pydantic import BaseModel


class DiceCoefficientSpec(BaseModel):
    name: str = "dice_coefficient"


class DiceLossSpec(BaseModel):
    name: str = "dice_loss"


class CceLossSpec(BaseModel):
    name: str = "cce_loss"


class CceDiceLossSpec(BaseModel):
    name: str = "cce_dice_loss"


class IouCoefficientSpec(BaseModel):
    name: str = "iou_coefficient"


class IouLossSpec(BaseModel):
    name: str = "iou_loss"


class LovaszLoss(BaseModel):
    name: str = "lovasz_loss"


class FocalLossSpec(BaseModel):
    gamma: float = 2
    alpha: float = 0.8
    name: str = "focal_loss"


class TverskyCoefficientSpec(BaseModel):
    alpha: float = .5
    beta: float = .5
    name: str = "tversky_coefficient"


class TverskyLossSpec(BaseModel):
    alpha: float = .5
    beta: float = .5
    name: str = "tversky_loss"


class FocalTverskyLossSpec(BaseModel):
    gamma: float = 2
    alpha: float = 0.5
    beta: float = 0.5
    name: str = "focal_tversky_loss"


class ComboLossSpec(BaseModel):
    alpha: float = 0.5
    ce_ratio: float = 0.5
    name: str = "combo_loss"
