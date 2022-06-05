import tensorflow as tf
import tensorflow.keras.backend as kb
from typing import Optional


class segmentation_loss:

    def __init__(
            self,
            n_class: int,
            background_index: Optional[int] = None
    ) -> None:
        """
        Set of loss functions and metrics for semantic segmentation.

        Args:
            n_class (int): Number of classes including background.
            background_index (int): Background index.
        """
        self.n_class = n_class
        self.background_index = background_index

    def remove_background(
            self,
            y: tf.Tensor
    ) -> tf.Tensor:
        """
        Removes background from true/predicted segmentation mask.

        Args:
            y (tf.Tensor): True/predicted segmentation mask.

        Returns:
            True/predicted segmentation mask without background.
        """
        if self.background_index is not None:
            idx = list(range(self.n_class))
            idx.remove(self.background_index)
            return tf.gather(y, idx, axis=-1)
        else:
            return y

    def dice_coefficient(
            self,
            y_actual: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Dice coefficient.

        Args:
            y_actual (tf.Tensor): True segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.

        Returns:
            Dice coefficient.
        """
        y_actual_ = self.remove_background(y_actual)
        y_pred_ = self.remove_background(y_pred)
        intersection = kb.sum(tf.cast(y_actual_, 'float32') * y_pred_)
        masks_sum = kb.sum(tf.cast(y_actual_, 'float32')) + kb.sum(y_pred_)
        return (2 * intersection + kb.epsilon()) / (masks_sum + kb.epsilon())

    def dice_loss(
            self,
            y_actual: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Dice loss.

        Args:
            y_actual (tf.Tensor): True segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.

        Returns:
            Dice loss.
        """
        return 1 - self.dice_coefficient(y_actual, y_pred)

    def CCE_loss(
            self,
            y_actual: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        CCE (categorical cross-entropy) loss.

        Args:
            y_actual (tf.Tensor): True segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.

        Returns:
            CCE (categorical cross-entropy) loss.
        """
        y_actual_ = self.remove_background(y_actual)
        y_pred_ = self.remove_background(y_pred)
        return kb.sum(kb.categorical_crossentropy(tf.cast(y_actual_, 'float32'), tf.cast(y_pred_, 'float32')))

    def CCE_dice_loss(
            self,
            y_actual: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        CCE-Dice loss.

        Args:
            y_actual (tf.Tensor): True segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.

        Returns:
            CCE-Dice loss.
        """
        return self.CCE_loss(y_actual, y_pred) + self.dice_loss(y_actual, y_pred)

    def IoU_coefficient(
            self,
            y_actual: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        IoU coefficient.

        Args:
            y_actual (tf.Tensor): True segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.

        Returns:
            IoU coefficient.
        """
        y_actual = self.remove_background(y_actual)
        y_pred = self.remove_background(y_pred)
        intersection = tf.cast(kb.sum(y_actual * y_pred), 'float32')
        union = tf.cast(kb.sum(y_actual) + kb.sum(y_pred), 'float32') - intersection
        return (intersection + kb.epsilon()) / (union + kb.epsilon())

    def IoU_loss(
            self,
            y_actual: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        IoU loss.

        Args:
            y_actual (tf.Tensor): True segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.

        Returns:
            IoU loss.
        """
        return 1 - self.IoU_coefficient(y_actual, y_pred)
