import tensorflow as tf
import tensorflow.keras.backend as kb
from typing import Optional
from platypus.utils.lovasz_softmax import LovaszSoftmaxLoss as LSL


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
        return kb.sum(kb.categorical_crossentropy(tf.cast(y_actual, 'float32'), tf.cast(y_pred, 'float32')))

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
        y_actual_ = self.remove_background(y_actual)
        y_pred_ = self.remove_background(y_pred)
        intersection = kb.sum(tf.cast(y_actual_, 'float32') * y_pred_)
        union = kb.sum(tf.cast(y_actual_, 'float32')) + kb.sum(y_pred_) - intersection
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

    def focal_loss(
        self,
        y_actual: tf.Tensor,
        y_pred: tf.Tensor,
        gamma: Optional[float] = 2,
        alpha: Optional[float] = 0.8
            ) -> tf.Tensor:
        """Calculates the focal loss using the categorical cross-entropy in the background.

        Args:
            y_actual (tf.Tensor): True segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.
            gamma (float): The higher the gamma is the more focues is shifted towards uncertain predictions.
            alpha (float): Weight parameter.

        Returns:
            focal_loss.
        """
        # TODO Should alpha be kept? To discuss.
        CEE = self.CCE_loss(y_actual, y_pred)
        pt = kb.exp(-CEE)
        return CEE*alpha*(1-pt)**gamma

    def tversky_coefficient(
        self,
        y_actual: tf.Tensor,
        y_pred: tf.Tensor,
        alpha: Optional[float] = .5,
        beta: Optional[float] = .5
            ) -> tf.Tensor:
        y_actual = tf.cast(self.remove_background(y_actual), "float32")
        y_pred = self.remove_background(y_pred)

        TP = kb.sum((y_actual * y_pred))
        FP = kb.sum(((1 - y_actual) * y_pred))
        FN = kb.sum((y_actual * (1 - y_pred)))

        tversky_coefficient = (TP + kb.epsilon()) / (TP + alpha*FP + beta*FN + kb.epsilon())  

        return tversky_coefficient

    def tversky_loss(
        self,
        y_actual: tf.Tensor,
        y_pred: tf.Tensor,
        alpha: Optional[float] = .5,
        beta: Optional[float] = .5
            ) -> tf.Tensor:
        return 1 - self.tversky_coefficient(y_actual, y_pred, alpha, beta)

    def focal_tversky_loss(
        self,
        y_actual: tf.Tensor,
        y_pred: tf.Tensor,
        gamma: Optional[float] = 2,
        alpha: Optional[float] = .5,
        beta: Optional[float] = .5
            ) -> tf.Tensor:
        tversky_coefficient = self.tversky_coefficient(y_actual, y_pred, alpha, beta)
        return (1 - tversky_coefficient) ** gamma

    def combo_loss(
        self,
        y_actual: tf.Tensor,
        y_pred: tf.Tensor,
        alpha: Optional[float] = .5,
        ce_ratio: Optional[float] = .5
            ) -> tf.Tensor:
        """Calculates the combo loss, being the combination od dice loss and false-negatives, false-positives penalization.

        Args:
            y_actual (tf.Tensor): True segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.
            alpha (float): Alpha < 0.5 penalizes FP more, otherwise the FN are penalized drastically.
            ce_ratio (float): Weighted contribution of modified CE loss compared to the Dice loss.

        Returns:
            combo_loss."""
        y_actual = tf.cast(self.remove_background(y_actual), "float32")
        y_pred = self.remove_background(y_pred)

        dice = self.dice_loss(y_actual, y_pred)

        y_pred = kb.clip(y_pred, kb.epsilon(), 1.0 - kb.epsilon())
        out = - (alpha * ((y_actual * kb.log(y_pred)) + ((1 - alpha) * (1.0 - y_actual) * kb.log(1.0 - y_pred))))

        weighted_ce = kb.mean(out, axis=-1)

        combo_loss = kb.mean((ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice))  # TODO Check
        return combo_loss

    def lovasz_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculates the Lovasz loss.

        Args:
            y_actual (tf.Tensor): True segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.

        Returns:
            lovasz_loss."""
        y_actual = tf.cast(self.remove_background(y_actual), tf.float64)
        y_pred = tf.cast(self.remove_background(y_pred), tf.float64)

        # TODO Think about the efficiency, it is said to yield better results though.
        # Calculated for each image separately.
        losses = tf.map_fn(LSL().lovasz_softmax_batch, (y_pred, y_actual), dtype=tf.float64)
        loss = tf.reduce_mean(losses)
        return lovasz_loss
