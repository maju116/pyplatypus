import pytest
import numpy as np
from platypus.segmentation.loss import *
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
             'CCE_dice_loss': 7.2127518447904855
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
    assert np.allclose(sl.CCE_loss(input_1, input_2).numpy(), output['CCE_loss'])
    assert np.allclose(sl.CCE_dice_loss(input_1, input_2).numpy(), output['CCE_dice_loss'])
