from pyplatypus.segmentation.loss_functions import SegmentationLoss
from tensorflow import Tensor


class LossGetter:
    def __init__(self, n_class, background_index, input_dict):
        self.loss_method = getattr(SegmentationLoss(n_class, background_index), input_dict.get("name"))
        input_dict.pop("name")
        self.loss_parameters = input_dict


class DiceCoefficientGetter(LossGetter):
    def dice_coefficient(self, x: Tensor, y: Tensor):
        return self.loss_method(x, y, **self.loss_parameters)


class DiceLossGetter(LossGetter):
    def dice_loss(self, x: Tensor, y: Tensor):
        return self.loss_method(x, y, **self.loss_parameters)


class CceLossGetter(LossGetter):
    def cce_loss(self, x: Tensor, y: Tensor):
        return self.loss_method(x, y, **self.loss_parameters)


class CceDiceLossGetter(LossGetter):
    def cce_dice_loss(self, x: Tensor, y: Tensor):
        return self.loss_method(x, y, **self.loss_parameters)


class IouCoefficientGetter(LossGetter):
    def iou_coefficient(self, x: Tensor, y: Tensor):
        return self.loss_method(x, y, **self.loss_parameters)


class IouLossGetter(LossGetter):
    def iou_loss(self, x: Tensor, y: Tensor):
        return self.loss_method(x, y, **self.loss_parameters)


class FocalLossGetter(LossGetter):
    def focal_loss(self, x: Tensor, y: Tensor):
        return self.loss_method(x, y, **self.loss_parameters)


class TverskyCoefficientGetter(LossGetter):
    def tversky_coefficient(self, x: Tensor, y: Tensor):
        return self.loss_method(x, y, **self.loss_parameters)


class TverskyLossGetter(LossGetter):
    def tversky_loss(self, x: Tensor, y: Tensor):
        return self.loss_method(x, y, **self.loss_parameters)


class FocalTverskyLossGetter(LossGetter):
    def focal_tversky_loss(self, x: Tensor, y: Tensor):
        return self.loss_method(x, y, **self.loss_parameters)
