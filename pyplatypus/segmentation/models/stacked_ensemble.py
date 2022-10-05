from tensorflow.keras.layers import (
    SeparableConv2D, BatchNormalization, MaxPool2D, Dropout, Conv2DTranspose,
    Concatenate, Cropping2D, Resizing, Average, Add, Conv2D, SpatialDropout2D,
    UpSampling2D
)
from tensorflow.keras import activations as KRACT
from tensorflow.keras.backend import int_shape
from tensorflow.keras import Model, Input
import tensorflow as tf
from typing import Tuple, Any, Optional, Union, List


class semantic_segmentation_ensembler:

    def __init__(
            self,
            models: List[tf.keras.Model],
            **kwargs
    ) -> None:
        """Creates semantic segmentation stacked ensemble model architecture.

        Parameters
        ----------
        models : List[tf.keras.Model]
            List with semantic segmentation models.
        """
        self.models = models
        self.model = self.build_model()

    def build_model(self) -> tf.keras.Model:
        """
        Creates a semantic segmentation stacked ensemble architecture.

        Returns
        -------
        model:
            Semantic segmentation stacked ensemble model.
        """
        for i in range(len(self.models)):
            model = self.models[i]
            for layer in model.layers:
                layer.trainable = False
                layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name
        ensemble_visible = [model.input for model in self.models]
        ensemble_outputs = [model.output for model in self.models]
        # ToDo: reshape outputs to the same HxW
        merge = Concatenate()(ensemble_outputs)
        # ToDo: Add Conv2d and Pool2d
        model = Model(inputs=ensemble_visible, outputs=merge)
        return model
