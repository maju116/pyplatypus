from pyplatypus.segmentation.loss_functions import SegmentationLoss


class LossGetter(n_class, background_index, input_dict):
    def _init__(self, n_class, background_index, input_dict):
        self.loss_method = getattr(SegmentationLoss(n_class, background_index), input_dict.get("name"))
        
        self.loss_parameters = 

def get_loss_method(n_class, background_index, input_dict):
    loss_method = getattr(SegmentationLoss(n_class, background_index), input_dict.get_name)


class 
