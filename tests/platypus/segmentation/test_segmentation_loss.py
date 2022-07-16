import pytest
import numpy as np
from platypus.segmentation.loss import segmentation_loss
import tensorflow as tf


@pytest.mark.parametrize(
    "n_class, background_index, input_1, input_2, output",
    [
        (4, None,
         tf.constant(np.array([0, 0, 0, 1,
                               1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 1, 0]).reshape((1, 2, 2, 4))),
         tf.constant(np.array([0.3, 0.2, 0.4, 0.1,
                               0.2, 0.3, 0.2, 0.3,
                               0.6, 0.4, 0, 0,
                               0.1, 0.1, 0.2, 0.6]).reshape((1, 2, 2, 4)), tf.float32),
         {
             'input_1_remove_background': tf.constant(np.array([0, 0, 0, 1,
                                                                1, 0, 0, 0,
                                                                0, 1, 0, 0,
                                                                0, 0, 1, 0]).reshape((1, 2, 2, 4))),
             'input_2_remove_background': tf.constant(np.array([0.3, 0.2, 0.4, 0.1,
                                                                0.2, 0.3, 0.2, 0.3,
                                                                0.6, 0.4, 0, 0,
                                                                0.1, 0.1, 0.2, 0.6]).reshape((1, 2, 2, 4)), tf.float32),
             'dice_coefficient': 0.22500015520951436,
             'dice_loss': 0.7749998447904857,
             'CCE_loss': 6.437752,
             'CCE_dice_loss': 7.2127518447904855,
             'IoU_coefficient': 0.12676057,
             'IoU_loss': 0.87323943
         }
         ),
        (4, 1,
         tf.constant(np.array([0, 0, 0, 1,
                               1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 1, 0]).reshape((1, 2, 2, 4))),
         tf.constant(np.array([0.3, 0.2, 0.4, 0.1,
                               0.2, 0.3, 0.2, 0.3,
                               0.6, 0.4, 0, 0,
                               0.1, 0.1, 0.2, 0.6]).reshape((1, 2, 2, 4)), tf.float32),
         {
             'input_1_remove_background': tf.constant(np.array([0, 0, 1,
                                                                1, 0, 0,
                                                                0, 0, 0,
                                                                0, 1, 0]).reshape((1, 2, 2, 3))),
             'input_2_remove_background': tf.constant(np.array([0.3, 0.4, 0.1,
                                                                0.2, 0.2, 0.3,
                                                                0.6, 0, 0,
                                                                0.1, 0.2, 0.6]).reshape((1, 2, 2, 3)), tf.float32),
             'dice_coefficient': 0.16666669,
             'dice_loss': 0.8333333134651184,
             'CCE_loss': 6.437752,
             'CCE_dice_loss': 7.271085313465118,
             'IoU_coefficient': 0.090909116,
             'IoU_loss': 0.9090908840298653
         }
         )
    ]
)
def test_segmentation_loss(n_class, background_index, input_1, input_2, output):
    sl = segmentation_loss(n_class=n_class, background_index=background_index)
    assert (sl.remove_background(input_1) == output['input_1_remove_background']).numpy().all()
    assert (sl.remove_background(input_2) == output['input_2_remove_background']).numpy().all()
    assert np.allclose(sl.dice_coefficient(input_1, input_2).numpy(), output['dice_coefficient'])
    assert np.allclose(sl.dice_loss(input_1, input_2).numpy(), output['dice_loss'])
    assert np.allclose(sl.cce_loss(input_1, input_2).numpy(), output['CCE_loss'])
    assert np.allclose(sl.cce_dice_loss(input_1, input_2).numpy(), output['CCE_dice_loss'])
    assert np.allclose(sl.iou_coefficient(input_1, input_2).numpy(), output['IoU_coefficient'])
    assert np.allclose(sl.iou_loss(input_1, input_2).numpy(), output['IoU_loss'])


class TestSegmentationLoss:
    sl = segmentation_loss(n_class=2, background_index=None)  
    y_actual = tf.constant(
        np.array([
            0, 0, 0, 1,
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0
            ]).reshape((2, 2, 2, 2)))

    y_pred = tf.constant(
        np.array([
            0.3, 0.2, 0.4, 0.1,
            0.2, 0.3, 0.2, 0.3,
            0.6, 0.4, 0, 0,
            0.1, 0.1, 0.2, 0.6
            ]).reshape((2, 2, 2, 2)), tf.float32)

    losses_path = 'platypus.segmentation.loss.segmentation_loss'

    def test_focal_loss(self, mocker):
        gamma = 1
        alpha = 1
        CEE = 1
        mocker.patch(self.losses_path + ".cce_loss", return_value=tf.constant([CEE], dtype=tf.float32))
        result = self.sl.focal_loss(self.y_actual, self.y_pred, gamma=gamma, alpha=alpha)
        assert result == alpha*(1-np.exp(-CEE))**gamma

    def test_extract_confusion_matrix(self):
        y_actual = tf.constant(
            np.array([
                0, 0, 0, 1,
                1, 0, 0, 0
                ]).reshape((1, 2, 2, 2)), tf.float32)

        y_pred = tf.constant(
            np.array([
                .5, 0, 0, 0,
                .5, 0, 0, 0
                ]).reshape((1, 2, 2, 2)), tf.float32)

        result = self.sl._extract_confusion_matrix(y_actual, y_pred)
        assert result == (tf.constant(.5), tf.constant(.5), tf.constant(1.5))

    def test_tversky_coefficient(self, mocker):
        alpha = 1
        beta = 1
        mocker.patch(self.losses_path + "._extract_confusion_matrix", return_value=(1, 1, 1))
        result = self.sl.tversky_coefficient(self.y_actual, self.y_pred, alpha=alpha, beta=beta)
        reference = 1 / (1 + alpha + beta)
        assert np.allclose(result, reference) 

    def test_tversky_loss(self, mocker):
        mocked_coefficient = .5
        mocker.patch(self.losses_path + ".tversky_coefficient", return_value=mocked_coefficient)
        result = self.sl.tversky_loss(self.y_actual, self.y_pred)
        assert result == 1 - mocked_coefficient

    def test_focal_tversky_loss(self, mocker):
        gamma = 2
        mocked_coefficient = .5
        mocker.patch(self.losses_path + ".tversky_coefficient", return_value=mocked_coefficient)
        result = self.sl.focal_tversky_loss(self.y_actual, self.y_pred)
        assert result == (1-mocked_coefficient)**gamma

    def test_combo_loss(self, mocker):
        y_actual = tf.constant(
            np.array([
                0, 0, 0, 1,
                1, 0, 0, 0
                ]).reshape((1, 2, 2, 2)), tf.float32)

        y_pred = tf.constant(
            np.array([
                .5, 0, 0, 0,
                .5, 0, 0, 0
                ]).reshape((1, 2, 2, 2)), tf.float32)

        alpha = 0
        ce_ratio = 0
        mocked_dice = 1
        mocker.patch(self.losses_path + ".dice_loss", return_value=mocked_dice)
        result = self.sl.combo_loss(y_actual, y_pred, alpha=alpha, ce_ratio=ce_ratio)
        assert result == -1

    def test_lovasz_loss(self, mocker):
        batch_losses = tf.constant([1, 0], dtype=tf.float64)
        mocker.patch("platypus.utils.lovasz_softmax.LovaszSoftmaxLoss.lovasz_softmax_batch", return_value=batch_losses)
        result = self.sl.lovasz_loss(self.y_actual, self.y_pred)
        assert result == np.mean(batch_losses)
