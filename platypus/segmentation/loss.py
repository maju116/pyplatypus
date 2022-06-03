import tensorflow as tf
import tensorflow.keras.backend as kb


def dice_coefficient(
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
    intersection = tf.cast(kb.sum(y_pred * y_actual), 'float32')
    masks_sum = tf.cast(kb.sum(y_pred) + kb.sum(y_actual), 'float32')
    return (2 * intersection + kb.epsilon()) / (masks_sum + kb.epsilon())


def dice_loss(
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
    return 1 - dice_coefficient(y_actual, y_pred)


