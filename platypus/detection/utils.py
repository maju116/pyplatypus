import numpy as np
from typing import Tuple, List, Union, Optional
import itertools

class yolo3_predict:

    def __init__(
            self,
            predictions: np.ndarray,
            anchors: List[List[Tuple]],
            labels: List[str],
            obj_threshold: float = 0.6,
            nms: bool = True,
            nms_threshold: float = 0.6,
            correct_hw: bool = False,
            image_h: Optional[int] = None,
            image_w: Optional[int] = None
    ) -> None:
        """
        Transforms Yolo3 model predictions into bounding boxes.

        Args:
            predictions (np.ndarray): Yolo3 model predictions.
            anchors (List[List[Tuple]]): Prediction anchors.
            labels (List[str]): Character vector containing class labels.
            obj_threshold (float): Minimum objectness score. Must be in range `[0, 1]`. All boxes with objectness score less than `obj_threshold` will be filtered out.
            nms (bool): Logical. Should `Non-Maximum-Suppression` be applied.
            nms_threshold (float): `Non-Maximum-Suppression` threshold.
            correct_hw (bool): Logical. Should height/width rescaling of bounding boxes be applied. If `TRUE` `xmin/xmax` coordinates are multiplied by `image_w` and `ymin/ymax` coordinates are multiplied by `image_h`.
            image_h (Optional[int]): Rescaling factor for `ymin/ymax` box coordinates.
            image_w (Optional[int]): Rescaling factor for `xmin/xmax` box coordinates.
        """
        self.predictions = predictions
        self.anchors = anchors
        self.labels = labels
        self.n_class = len(labels)
        self.obj_threshold = obj_threshold
        self.nms = nms
        self.nms_threshold = nms_threshold
        self.correct_hw = correct_hw
        self.image_h = image_h
        self.image_w = image_w
        self.anchors_per_grid = len(anchors[0])
        self.n_images = self.predictions[0].shape[0]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def logit(x):
        return np.log(x / (1 - x))

    def get_boxes(
            self
    ):
        return self.predictions

    def transform_boxes(
            self
    ):
        return 33

    def transform_boxes_for_grid(
            self,
            grid_predictions: np.ndarray,
            grid_anchors: List
    ) -> List:
        grid_h = grid_predictions.shape[0]
        grid_w = grid_predictions.shape[1]
        grid_dims = list(itertools.product(range(grid_w), range(grid_h)))
        boxes = []
        for wh in grid_dims:
            row = wh[1]
            col = wh[0]
            for count, anchor in enumerate(grid_anchors):
                box_data = grid_predictions[row, col, count, :]
                if self.sigmoid(box_data[4]) > self.obj_threshold:
                    box_data[0] = (self.sigmoid(box_data[0]) + col) / grid_w
                    box_data[1] = (self.sigmoid(box_data[1]) + row) / grid_h
                    box_data[2] = anchor[0] * np.exp(box_data[2])
                    box_data[3] = anchor[1] * np.exp(box_data[3])
                    box_data[4] = self.sigmoid(box_data[4])
                    box_data[5:] = box_data[4] * self.sigmoid(box_data[5:])
                    box_data[5:] = (box_data[5:] == max(box_data[5:])) & (
                                box_data[5:] > self.obj_threshold)
                    xmin = box_data[0] - box_data[2] / 2
                    ymin = box_data[1] - box_data[3] / 2
                    xmax = box_data[0] + box_data[2] / 2
                    ymax = box_data[1] + box_data[3] / 2
                    box_data[:4] = (xmin, ymin, xmax, ymax)
                    boxes.append(box_data)
        return boxes
