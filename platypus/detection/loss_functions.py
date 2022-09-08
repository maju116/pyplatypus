import tensorflow as tf
import tensorflow.keras.backend as kb
from typing import Optional, List, Tuple


class yolo3_loss:

    def __init__(
            self,
            anchors: List[List[Tuple]],
            n_class: int = 80,
            nonobj_threshold: float = 0.5,
            bbox_lambda: float = 1,
            obj_lambda: float = 1,
            noobj_lambda: float = 1,
            class_lambda: float = 1,
            class_weights = [1] * 80
    ) -> None:
        """
        Set of loss functions and metrics for Yolo3 object detection.

        Args:
            anchors (List[List[Tuple, Tuple]]): Prediction anchors.
            n_class (int): Number of classes including background.
        """
        self.anchors = anchors
        self.n_class = n_class
        self.nonobj_threshold = nonobj_threshold
        self.bbox_lambda = bbox_lambda
        self.obj_lambda = obj_lambda
        self.noobj_lambda = noobj_lambda
        self.class_lambda = class_lambda
        self.class_weights = class_weights

    def transform_boxes_tf(
            self,
            predictions: tf.Tensor,
            grid_anchors: List[Tuple],
            transform_proba: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Transforms Yolo3 predictions into valid box coordinates/scores.

        Args:
            predictions (tf.Tensor): Yolo3 model predictions (from one grid).
            grid_anchors (List[Tuple]): Prediction anchors (for one grid).
            transform_proba (bool): Logical. Should the score/class probabilities be transformed.

        Returns:
            Transformed bounding box coordinates/scores.
        """
        grid_h = predictions.shape.as_list()[1]
        grid_w = predictions.shape.as_list()[2]
        box_split = tf.split(predictions, num_or_size_splits=(1, 1, 1, 1, 1, self.n_class), axis=-1)
        box_x = kb.sigmoid(box_split[0])
        box_y = kb.sigmoid(box_split[1])
        box_w = box_split[2]
        box_h = box_split[3]
        score = kb.sigmoid(box_split[4]) if transform_proba else box_split[4]
        class_probs = kb.sigmoid(box_split[5]) if transform_proba else box_split[5]

        grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        grid_col = tf.expand_dims(tf.expand_dims(grid[0], axis=-1), axis=-1)
        grid_row = tf.expand_dims(tf.expand_dims(grid[1], axis=-1), axis=-1)

        box_x = (box_x + tf.cast(grid_col, tf.float32)) / tf.cast(grid_w, tf.float32)
        box_y = (box_y + tf.cast(grid_row, tf.float32)) / tf.cast(grid_h, tf.float32)

        anchors_tf = tf.constant(grid_anchors, tf.float32)
        anchors_tf = tf.expand_dims(tf.expand_dims(anchors_tf, axis=0), axis=0)
        anchors_tf = tf.split(anchors_tf, num_or_size_splits=(1, 1), axis=-1)
        box_w = kb.exp(box_w) * anchors_tf[0]
        box_h = kb.exp(box_h) * anchors_tf[1]

        bbox = kb.concatenate((box_x, box_y, box_w, box_h), axis=-1)
        return bbox, score, class_probs

    @staticmethod
    def transform_box_to_min_max(
            box: tf.Tensor
    ) -> tf.Tensor:
        """
        Transforms boxes to min/max coordinates.

        Args:
            box: Boxes coordinates.

        Returns:
            Boxes min/max coordinates.
        """
        box_xmin = tf.expand_dims(box[:, :, :, :, 0] - box[:, :, :, :, 2] / 2, axis=-1)
        box_ymin = tf.expand_dims(box[:, :, :, :, 1] - box[:, :, :, :, 3] / 2, axis=-1)
        box_xmax = tf.expand_dims(box[:, :, :, :, 0] + box[:, :, :, :, 2] / 2, axis=-1)
        box_ymax = tf.expand_dims(box[:, :, :, :, 1] + box[:, :, :, :, 3] / 2, axis=-1)
        return kb.concatenate((box_xmin, box_ymin, box_xmax, box_ymax), axis=-1)

    @staticmethod
    def calculate_iou(
            pred_boxes: tf.Tensor,
            true_boxes: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculates boxes IoU.

        Args:
            pred_boxes (tf.Tensor): Tensor of predicted coordinates.
            true_boxes (tf.Tensor): Tensor of true coordinates.

        Returns:
            IoU between true and predicted boxes.
        """
        intersection_w = tf.maximum(tf.minimum(pred_boxes[:, :, :, :, 2], true_boxes[:, :, :, :, 2]) -
                                    tf.maximum(pred_boxes[:, :, :, :, 0], true_boxes[:, :, :, :, 0]), 0)
        intersection_h = tf.maximum(tf.minimum(pred_boxes[:, :, :, :, 3], true_boxes[:, :, :, :, 3]) -
                                    tf.maximum(pred_boxes[:, :, :, :, 1], true_boxes[:, :, :, :, 1]), 0)
        intersection_area = intersection_w * intersection_h
        pred_boxes_area = (pred_boxes[:, :, :, :, 2] - pred_boxes[:, :, :, :, 0]) * (pred_boxes[:, :, :, :, 3] - pred_boxes[:, :, :, :, 1])
        true_boxes_area = (true_boxes[:, :, :, :, 2] - true_boxes[:, :, :, :, 0]) * (true_boxes[:, :, :, :, 3] - true_boxes[:, :, :, :, 1])
        return intersection_area / (pred_boxes_area + true_boxes_area - intersection_area)

    def get_max_boxes_iou(
            self,
            pred_boxes: tf.Tensor,
            true_boxes: tf.Tensor
    ) -> tf.Tensor:
        """
        ompares boxes by IoU.

        Args:
            pred_boxes (tf.Tensor): Tensor of predicted coordinates.
            true_boxes (tf.Tensor): Tensor of true coordinates.

        Returns:
            Max IoU between true and predicted boxes.
        """
        pred_boxes = tf.expand_dims(pred_boxes, axis=-2)
        true_boxes = tf.expand_dims(true_boxes, axis=0)
        new_shape = tf.broadcast_dynamic_shape(tf.shape(pred_boxes), tf.shape(true_boxes))
        pred_boxes = tf.broadcast_to(pred_boxes, new_shape)
        true_boxes = tf.broadcast_to(true_boxes, new_shape)
        return self.calculate_iou(pred_boxes, true_boxes)

    def yolo3_grid_loss(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            anchors: List[Tuple]
    ) -> tf.Tensor:
        """
        Calculates loss for one `Yolo3` grid.

        Args:
            y_true (tf.Tensor): Tensor of true coordinates/scores.
            y_pred (tf.Tensor): Tensor of predicted coordinates/scores.
            anchors (List[Tuple]): Prediction anchors (for one grid).

        Returns:
            Loss for one `Yolo3` grid.
        """
        true_boxes = self.transform_boxes_tf(y_true, anchors, self.n_class, transform_proba=False)
        pred_boxes = self.transform_boxes_tf(y_pred, anchors, self.n_class, transform_proba=True)
        true_boxes_min_max = self.transform_box_to_min_max(true_boxes[0])
        pred_boxes_min_max = self.transform_box_to_min_max(pred_boxes[0])

        bbox_scale = 2 - true_boxes[0][:, :, :, :, 2] * true_boxes[0][:, :, :, :, 3]
        obj_mask = tf.squeeze(true_boxes[1], axis=-1)
        bbox_loss = bbox_scale * obj_mask * tf.reduce_sum(tf.square(true_boxes[0] - pred_boxes[0]), axis=-1)

        max_iou = tf.map_fn(lambda x: tf.reduce_max(
            self.get_max_boxes_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2]. tf.bool))), axis=-1
        ), (pred_boxes_min_max, true_boxes_min_max, obj_mask), tf.float32)
        ignore_mask = tf.cast(max_iou < self.nonobj_threshold, tf.float32)
        obj_loss_bc = tf.keras.losses.BinaryCrossentropy(true_boxes[1], pred_boxes[1])
        obj_loss = obj_mask * obj_loss_bc
        noobj_loss = (1 - obj_mask) * obj_loss_bc * ignore_mask

        class_loss = 0
        for cls in range(self.n_class):
            current_class_true = tf.expand_dims(true_boxes[2][:, :, :, :, cls], axis=-1)
            current_class_false = 1 - current_class_true
            current_class = kb.concatenate((current_class_true, current_class_false), axis=-1)
            current_class_pred_true = tf.expand_dims(pred_boxes[2][:, :, :, :, cls], axis=-1)
            current_class_pred_false = 1 - current_class_pred_true
            current_class_pred = kb.concatenate((current_class_pred_true, current_class_pred_false), axis=-1)
            current_class_bc = tf.keras.losses.BinaryCrossentropy(current_class, current_class_pred)
            class_loss = class_loss + self.class_weights[cls] * current_class_bc
        class_loss = class_loss * obj_mask

        bbox_loss = self.bbox_lambda * tf.reduce_sum(bbox_loss, axis=(1, 2, 3))
        obj_loss = self.obj_lambda * tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        noobj_loss = self.noobj_lambda * tf.reduce_sum(noobj_loss, axis=(1, 2, 3))
        class_loss = self.class_lambda * tf.reduce_sum(class_loss, axis=(1, 2, 3))
        total_loss = bbox_loss + obj_loss + noobj_loss + class_loss
        return total_loss

    def yolo3_loss(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor
    ) -> List:
        """
        Generates `Yolo3` loss function.

        Agrs:
            y_true (tf.Tensor): Tensor of true coordinates/scores.
            y_pred (tf.Tensor): Tensor of predicted coordinates/scores.

        Returns:
            `Yolo3` loss function.
        """
        return [lambda yt, yp: self.yolo3_grid_loss(yt, yp, a) for yt, yp, a in zip(y_true, y_pred, self.anchors)]


