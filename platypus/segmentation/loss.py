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
    intersection = tf.cast(kb.sum(y_actual * y_pred), 'float32')
    masks_sum = tf.cast(kb.sum(y_actual) + kb.sum(y_pred), 'float32')
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


def BCE_dice_loss(
        y_actual: tf.Tensor,
        y_pred: tf.Tensor
) -> tf.Tensor:
    """
    BCE-Dice loss.

    Args:
        y_actual (tf.Tensor): True segmentation mask.
        y_pred (tf.Tensor): Predicted segmentation mask.

    Returns:
        BCE-Dice loss.
    """
    BCE = kb.binary_crossentropy(tf.cast(kb.sum(y_actual), 'float32'), tf.cast(kb.sum(y_pred), 'float32'))
    return BCE + dice_loss(y_actual, y_pred)


def IoU_coefficient(
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
    intersection = tf.cast(kb.sum(y_actual * y_pred), 'float32')
    union = tf.cast(kb.sum(y_actual) + kb.sum(y_pred), 'float32') - intersection
    return (intersection + kb.epsilon()) / (union + kb.epsilon())


def IoU_loss(
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
    return 1 - IoU_coefficient(y_actual, y_pred)
