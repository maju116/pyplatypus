import tensorflow as tf
import tensorflow.keras.backend as kb
from typing import Optional
from pyplatypus.utils.lovasz_softmax import LovaszSoftmaxLoss as LSL


class SegmentationLoss:
    """The class storing all the implemented loss functions and coefficients.

    Methods
    -------
    dice_coefficient(self, y_actual: tf.Tensor, y_pred: tf.Tensor)
        Calculates the dice coefficient.

    dice_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor)
        Calculates the dice loss, based on the dice coefficient.

    cce_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor)
        Calculates the CCE (categorical cross-entropy) loss.

    cce_dice_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor)
        Calculates the mixed CCE-Dice loss.

    iou_coefficient(self, y_actual: tf.Tensor, y_pred: tf.Tensor)
        Calculates the IoU (intersect-over-union) coefficient.

    iou_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor)
        Calculates the IoU (intersect-over-union) loss based on the IoU coefficient.

    focal_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor, gamma: Optional[float] = 2, alpha: Optional[float] = 0.8)
        Calculates the focal loss using the categorical cross-entropy in the background.

    tversky_coefficient(self, y_actual: tf.Tensor, y_pred: tf.Tensor, alpha: Optional[float] = .5, beta: Optional[float] = .5)
        Calculates the Tversky coefficient which takes under consideration the True/False positives and the False-Negatives.

    tversky_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor, alpha: Optional[float] = .5, beta: Optional[float] = .5)
        Calculates the Tversky loss by subtracting the Tversky coefficient from one.

    focal_tversky_loss(
        self, y_actual: tf.Tensor, y_pred: tf.Tensor, gamma: Optional[float] = 2,
        alpha: Optional[float] = .5, beta: Optional[float] = .5
        )
        Calculates the Tversky loss and but gives the opportunity to manipulate the relation
        between the Tversky coefficient and the loss via the gamma parameter.

    combo_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor, alpha: Optional[float] = .5, ce_ratio: Optional[float] = .5)
        Calculates the combo loss, being the combination od dice loss and false-negatives, false-positives penalization.

    lovasz_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor)
        Calculates the Lovasz loss.
    """

    def __init__(self, n_class: int, background_index: Optional[int] = None) -> None:
        """
        Initializes the class storing the essentials as the number of class and the positioning of
        the background within the class.

        Parameters
        ----------
        n_class: int
            Number of classes including background.
        background_index: int
            Background index.
        """
        self.n_class = n_class
        self.background_index = background_index

    def remove_background(self, y: tf.Tensor) -> tf.Tensor:
        """
        Removes background from true/predicted segmentation mask.

        Parameters
        ----------
        y: tf.Tensor
            True or predicted segmentation mask.

        Returns
        -------
        image_without_the_background: tf.Tensor
            True or predicted segmentation mask without background.
        """
        if self.background_index is not None:
            idx = list(range(self.n_class))
            idx.remove(self.background_index)
            image_without_the_background = tf.gather(y, idx, axis=-1)
            return image_without_the_background
        else:
            image_without_the_background = y
            return image_without_the_background

    def dice_coefficient(self, y_actual: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculates the dice coefficient.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.

        Returns
        -------
        dice_coefficient: tf.Tensor
            Dice coefficient.
        """
        y_actual_ = self.remove_background(y_actual)
        y_pred_ = self.remove_background(y_pred)
        intersection = kb.sum(tf.cast(y_actual_, 'float32') * y_pred_)
        masks_sum = kb.sum(tf.cast(y_actual_, 'float32')) + kb.sum(y_pred_)
        dice_coefficient = (2 * intersection + kb.epsilon()) / (masks_sum + kb.epsilon())
        return dice_coefficient

    def dice_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculates the dice loss, based on the dice coefficient.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.

        Returns
        -------
        dice_loss: tf.Tensor
            Dice loss.
        """
        dice_loss = 1 - self.dice_coefficient(y_actual, y_pred)
        return dice_loss

    def cce_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculates the CCE (categorical cross-entropy) loss.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.

        Returns
        -------
        cce_loss: tf.Tensor
            CCE (categorical cross-entropy) loss.
        """
        cce_loss = kb.mean(kb.categorical_crossentropy(tf.cast(y_actual, 'float32'), tf.cast(y_pred, 'float32')))
        return cce_loss

    def cce_dice_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculates the mixed CCE-Dice loss.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.

        Returns
        -------
            CCE-Dice loss.
        """
        cce_dice_loss = self.cce_loss(y_actual, y_pred) + self.dice_loss(y_actual, y_pred)
        return cce_dice_loss

    def iou_coefficient(self, y_actual: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculates the IoU (intersect-over-union) coefficient.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.

        Returns
        -------
        iou_coefficient: tf.Tensor
            IoU coefficient.
        """
        y_actual_ = self.remove_background(y_actual)
        y_pred_ = self.remove_background(y_pred)
        intersection = kb.sum(tf.cast(y_actual_, 'float32') * y_pred_)
        union = kb.sum(tf.cast(y_actual_, 'float32')) + kb.sum(y_pred_) - intersection
        iou_coefficient = (intersection + kb.epsilon()) / (union + kb.epsilon())
        return iou_coefficient

    def iou_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculates the IoU (intersect-over-union) loss based on the IoU coefficient.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.

        Returns
        -------
        iou_loss: tf.Tensor
            IoU loss.
        """
        iou_loss = 1 - self.iou_coefficient(y_actual, y_pred)
        return iou_loss

    def focal_loss(
        self, y_actual: tf.Tensor, y_pred: tf.Tensor, gamma: Optional[float] = 2, alpha: Optional[float] = 0.8
            ) -> tf.Tensor:
        """Calculates the focal loss using the categorical cross-entropy in the background.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.
        gamma: float
            The higher the gamma is the more focues is shifted towards uncertain predictions.
        alpha: float
            Weight parameter.

        Returns
        -------
        focal_loss: tf.Tensor
            Focal loss.
        """
        CCE_pixelwise = kb.categorical_crossentropy(tf.cast(y_actual, 'float32'), tf.cast(y_pred, 'float32'))
        pt = kb.exp(-CCE_pixelwise)
        focal_loss = kb.mean(CCE_pixelwise*alpha*(1-pt)**gamma)
        return focal_loss

    @staticmethod
    def _extract_confusion_matrix(y_actual: tf.Tensor, y_pred: tf.Tensor) -> tuple:
        """Calculates True Positives, False-Positives and False-Negatives.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.

        Returns
        -------
        statistics: tuple
            True-Positives, False-Positives, False-Negatives rates.
        """
        TP = kb.sum((y_actual * y_pred))
        FP = kb.sum(((1 - y_actual) * y_pred))
        FN = kb.sum((y_actual * (1 - y_pred)))
        statistics = TP, FP, FN
        return statistics

    def tversky_coefficient(
        self, y_actual: tf.Tensor, y_pred: tf.Tensor, alpha: Optional[float] = .5, beta: Optional[float] = .5
            ) -> tf.Tensor:
        """Calculates the Tversky coefficient which takes under consideration the True/False positives and the False-Negatives.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.

        Returns
        -------
        tversky_coefficient: tf.Tensor
            Tversky coefficient.
        """
        y_actual = tf.cast(self.remove_background(y_actual), "float32")
        y_pred = self.remove_background(y_pred)

        TP, FP, FN = self._extract_confusion_matrix(y_actual, y_pred)
        tversky_coefficient = (TP + kb.epsilon()) / (TP + alpha*FP + beta*FN + kb.epsilon())  

        return tversky_coefficient

    def tversky_loss(
        self, y_actual: tf.Tensor, y_pred: tf.Tensor, alpha: Optional[float] = .5, beta: Optional[float] = .5
            ) -> tf.Tensor:
        """Calculates the Tversky loss by subtracting the Tversky coefficient from one.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.

        Returns
        -------
        tversky_loss: (tf.Tensor)
            Tversky loss.
        """
        tversky_loss = 1 - self.tversky_coefficient(y_actual, y_pred, alpha, beta)
        return tversky_loss

    def focal_tversky_loss(
        self, y_actual: tf.Tensor, y_pred: tf.Tensor, gamma: Optional[float] = 2,
        alpha: Optional[float] = .5, beta: Optional[float] = .5
            ) -> tf.Tensor:
        """Calculates the Tversky loss and but gives the opportunity to manipulate the relation
        between the Tversky coefficient and the loss via the gamma parameter.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.

        Returns
        -------
        focal_tversky_loss: tf.Tensor
            Modified Tversky loss.
        """
        tversky_coefficient = self.tversky_coefficient(y_actual, y_pred, alpha, beta)
        focal_tversky_loss = (1 - tversky_coefficient) ** gamma
        return focal_tversky_loss

    def combo_loss(
        self, y_actual: tf.Tensor, y_pred: tf.Tensor, alpha: Optional[float] = .5, ce_ratio: Optional[float] = .5
            ) -> tf.Tensor:
        """Calculates the combo loss, being the combination od dice loss and false-negatives, false-positives penalization.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.
        alpha: float
            Alpha < 0.5 penalizes FP more, otherwise the FN are penalized drastically.
        ce_ratio: float
            Weighted contribution of modified CE loss compared to the Dice loss.

        Returns
        -------
        combo_loss: tf.Tensor
            Combo loss."""
        y_actual = tf.cast(self.remove_background(y_actual), "float32")
        y_pred = self.remove_background(y_pred)

        dice = self.dice_loss(y_actual, y_pred)

        y_pred = kb.clip(y_pred, kb.epsilon(), 1.0 - kb.epsilon())
        out = - (alpha * ((y_actual * kb.log(y_pred)) + ((1 - alpha) * (1.0 - y_actual) * kb.log(1.0 - y_pred))))

        weighted_ce = kb.mean(out, axis=-1)

        combo_loss = kb.mean((ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice))
        return combo_loss

    def lovasz_loss(self, y_actual: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculates the Lovasz loss.

        Parameters
        ----------
        y_actual: tf.Tensor
            True segmentation mask.
        y_pred: tf.Tensor
            Predicted segmentation mask.

        Returns
        -------
        lovasz_loss: tf.Tensor
            Lovasz loss

        See also
        --------
        LovaszSoftmaxLoss from pyplatypus.utils.lovasz_softmax
        """
        y_actual = tf.cast(self.remove_background(y_actual), tf.float64)
        y_pred = tf.cast(self.remove_background(y_pred), tf.float64)

        # TODO Think about the efficiency, it is said to yield better results though.
        # Calculated for each image separately.
        losses = tf.map_fn(LSL().lovasz_softmax_batch, (y_pred, y_actual), dtype=tf.float64)
        lovasz_loss = tf.reduce_mean(losses)
        return lovasz_loss
