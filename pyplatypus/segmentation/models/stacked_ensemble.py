from tensorflow.keras.layers import (
    SeparableConv2D, BatchNormalization, MaxPool2D, Dropout, Conv2DTranspose,
    Concatenate, Cropping2D, Resizing, Average, Add, Conv2D, SpatialDropout2D,
    UpSampling2D
)
from tensorflow.keras.models import clone_model
from tensorflow.keras import activations as KRACT
from tensorflow.keras.backend import int_shape
from tensorflow.keras import Model, Input
import tensorflow as tf
from typing import Tuple, Any, Optional, Union, List


class semantic_segmentation_ensembler:

    def __init__(
            self,
            submodels: List[tf.keras.Model],
            copy_submodels_weights: bool = True,
            freeze_submodels_weights: bool = True,
            **kwargs
    ) -> None:
        """Creates semantic segmentation stacked ensemble model architecture.

        Parameters
        ----------
        submodels : List[tf.keras.Model]
            List with semantic segmentation models.
        copy_submodels_weights : bool
            Should the stacked ensembler be initialized with submodels weights.
        freeze_submodels : bool
            Should the submodels weights be freezed (only if copy_submodels_weights == `True`).
        """
        self.submodels = submodels
        self.copy_submodels_weights = copy_submodels_weights
        self.freeze_submodels_weights = freeze_submodels_weights
        self.submodels = self.copy_submodels()
        self.model = self.build_model()

    def copy_submodels(self) -> List[tf.Tensor]:
        """
        Creates a copy of submodels.

        Returns:
            List of submodels.
        """
        submodels_copy = [clone_model(m) for m in self.submodels]
        if self.copy_submodels_weights:
            for mc, m in zip(submodels_copy, self.submodels):
                mc.set_weights(m.get_weights())
        return submodels_copy

    def build_model(self) -> tf.keras.Model:
        """
        Creates a semantic segmentation stacked ensemble architecture.

        Returns
        -------
        model:
            Semantic segmentation stacked ensemble model.
        """
        for i in range(len(self.submodels)):
            model = self.submodels[i]
            for layer in model.layers:
                layer.trainable = False
                layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name
        ensemble_visible = [model.input for model in self.submodels]
        ensemble_outputs = [model.output for model in self.submodels]
        # ToDo: reshape outputs to the same HxW
        merge = Concatenate()(ensemble_outputs)
        # ToDo: Add Conv2d and Pool2d
        model = Model(inputs=ensemble_visible, outputs=merge)
        return model
