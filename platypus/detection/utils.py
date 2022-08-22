import numpy as np
from typing import Tuple, List, Union, Optional
import itertools
import pandas as pd


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
        boxes = self.transform_boxes()
        nms_boxes = []
        for b in boxes:
            image_nms_boxes = []
            labels = np.argmax(b[:, 5:], axis=1)
            unique_lebels = list(set(labels))
            for l in unique_lebels:
                class_boxes = b[labels == l, :][:, :5]
                class_nms_boxes = self.class_nms(class_boxes)
                class_nms_boxes['label_id'] = l
                class_nms_boxes['label'] = self.labels[l]
                image_nms_boxes.append(class_nms_boxes)
            image_nms_boxes = pd.concat(image_nms_boxes)
            nms_boxes.append(image_nms_boxes)
        return nms_boxes

    def transform_boxes(
            self
    ):
        image_boxes = []
        for image_nr in range(self.n_images):
            current_preds = [im[image_nr, :, :, :, :] for im in self.predictions]
            current_boxes = []
            for i in range(3):
                grid_predictions = current_preds[i]
                grid_anchors = self.anchors[i]
                current_boxes.append(self.transform_boxes_for_grid(grid_predictions, grid_anchors))
            current_boxes = np.stack([item for sublist in current_boxes for item in sublist], axis=0)
            image_boxes.append(current_boxes)
        return image_boxes

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

    @staticmethod
    def check_boxes_intersect(box1, box2):
        x_intersect = box1[0] < box2[2] and box1[2] > box2[0]
        y_intersect = box1[1] < box2[3] and box1[3] > box2[1]
        return x_intersect and y_intersect

    def intersection_over_union(self, box1, box2):
        boxes_intersect = self.check_boxes_intersect(box1, box2)
        if boxes_intersect:
            intersection = (min(box1[2], box2[2]) - (box1[0] if box2[0] < box1[0] else box2[0])) * \
                           (min(box1[3], box2[3]) - (box1[1] if box2[1] < box1[1] else box2[1]))
            union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
                    (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
            iou = intersection / union
        else:
            iou = 0
        return iou

    def class_nms(self, boxes):
        if len(boxes) > 1:
            comb = list(itertools.product(range(len(boxes)), range(len(boxes))))
            comb = [c for c in comb if c[0] < c[1]]
            ious = [self.intersection_over_union(boxes[c[0]], boxes[c[1]]) for c in comb]
            bb = np.stack([np.array([c[0], c[1], iou]) for c, iou in zip(comb, ious)], axis=0)
            bb = pd.DataFrame(bb, columns=['box1', 'box2', 'iou'])
            p1 = pd.DataFrame(boxes[:, 4], columns=['p1'])
            p1['box1'] = range(len(boxes))
            p2 = pd.DataFrame(boxes[:, 4], columns=['p2'])
            p2['box2'] = range(len(boxes))
            bb = pd.merge(bb, p1, on='box1', how='left')
            bb = pd.merge(bb, p2, on='box2', how='left')
            boxes_df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax', 'p'])
            boxes_df['box'] = range(len(boxes))
            boxes_to_remove = []
            for b in range(len(bb)):
                if bb.loc[b, 'iou'] > self.nms_threshold:
                    if bb.loc[b, 'p1'] < bb.loc[b, 'p2']:
                        boxes_to_remove.append(bb.loc[b, 'box1'])
                    else:
                        boxes_to_remove.append(bb.loc[b, 'box2'])
            boxes_to_remove = list(set(boxes_to_remove))
            class_nms_boxes = boxes_df[~boxes_df.box.isin(boxes_to_remove)].drop(columns=['box'])
        else:
            class_nms_boxes = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax', 'p'])
        return class_nms_boxes
