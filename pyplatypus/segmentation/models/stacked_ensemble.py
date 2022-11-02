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


class stacked_ensembler:

    def __init__(
            self,
            submodels: List[tf.keras.Model],
            copy_submodels_weights: bool,
            freeze_submodels_weights: bool,
            net_h: int,
            net_w: int,
            n_class: Optional[int] = 2,
            filters: Optional[int] = 16,
            kernel_size: Optional[Tuple[int, int]] = (3, 3),
            dropout: Optional[float] = 0.1,
            batch_normalization: Optional[bool] = True,
            kernel_initializer: Optional[str] = "he_normal",
            use_separable_conv2d: Optional[bool] = True,
            use_spatial_dropout2d: Optional[bool] = True,
            activation_layer: Optional[str] = "relu",
            u_net_conv_block_width: Optional[int] = 2,
            **kwargs
    ) -> None:
        """Creates semantic segmentation stacked ensemble model architecture.

        Parameters
        ----------
        submodels : List[tf.keras.Model]
            List with semantic segmentation models.
        net_h : int
            Input layer height.
        net_w : int
            Input layer width.
        filters: int
            Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
        kernel_size: Tuple[int, int]
            An integer or tuple of 2 integers, specifying the width and height of the 2D convolution window.
            It is allowed to be a single integer to specify the same value for all spatial dimensions.
        u_net_conv_block_width: int
            Controls the amount of convolutional layers in the block, by default 2
        grayscale : bool
        copy_submodels_weights : bool
            Should the stacked ensembler be initialized with submodels weights.
        freeze_submodels : bool
            Should the submodels weights be freezed (only if copy_submodels_weights == `True`).
        """
        self.submodels = submodels
        self.copy_submodels_weights = copy_submodels_weights
        self.freeze_submodels_weights = freeze_submodels_weights
        self.net_h = net_h
        self.net_w = net_w
        self.n_class = n_class
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.kernel_initializer = kernel_initializer
        self.use_separable_conv2d = use_separable_conv2d
        self.use_spatial_dropout2d = use_spatial_dropout2d
        self.activation_layer = activation_layer
        self.u_net_conv_block_width = u_net_conv_block_width
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

    def dropout_layer(self):
        """Creates the dropout layer of the preferred kind.

        Returns
        -------
        dropout_layer: KerasTensor
            Dropout created by the chosen method.
        """
        if self.use_spatial_dropout2d:
            dropout_layer = SpatialDropout2D(rate=self.dropout)
        else:
            dropout_layer = Dropout(rate=self.dropout)
        return dropout_layer

    def convolutional_layer(
            self, filters: int, kernel_size: Tuple[int, int], activation: Optional[str] = "linear"
    ) -> Union[SeparableConv2D, Conv2D]:
        """
        Returns the convolutional layer of the demanded type.

        Parameters
        ----------
        filters: int
            Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
        kernel_size: Tuple[int, int])
            An integer or tuple of 2 integers, specifying the width and height of the 2D convolution window.
            If single integer is supplied the same value is used for all spatial dimensions.
        activation: Optional[str]
            Activation function name.

        Returns
        -------
        convolutional layer.
        """
        if self.use_separable_conv2d:
            convolutional_layer = SeparableConv2D(
                filters=filters, kernel_size=kernel_size, padding="same",
                kernel_initializer=self.kernel_initializer, activation=activation)
        else:
            convolutional_layer = Conv2D(
                filters=filters, kernel_size=kernel_size, padding="same",
                kernel_initializer=self.kernel_initializer, activation=activation)
        return convolutional_layer

    def activation(self):
        """Creates the layer applying the specified activation function.

        Returns
        -------
        activation_layer: function
            Layer later used to apply the chosen activation function.
        """
        activation_layer = getattr(KRACT, self.activation_layer)
        return activation_layer

    def multiple_conv2d(
            self,
            input: tf.Tensor
    ) -> tf.Tensor:
        """
        Creates a multiple convolutional U-Net block.

        Parameters
        ----------
        input: tf.Tensor
            Model or layer object.

        Returns
        -------
        input:
            Multiple convolutional block of the model.
        """
        for i in range(self.u_net_conv_block_width):
            input = self.convolutional_layer(self.filters, self.kernel_size)(input)
            if self.batch_normalization:
                input = BatchNormalization()(input)
            input = self.activation()(input)
        return input

    def build_model(self) -> tf.keras.Model:
        """
        Creates a semantic segmentation stacked ensemble architecture.

        Returns
        -------
        model:
            Semantic segmentation stacked ensemble model.
        """
        for i in range(len(self.submodels)):
            submodel = self.submodels[i]
            if self.copy_submodels_weights and self.freeze_submodels_weights:
                for layer in submodel.layers:
                    layer.trainable = False
                    layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name
        inputs = [submodel.input for submodel in self.submodels]
        outputs = [submodel.output for submodel in self.submodels]
        outputs = [Resizing(height=self.net_h, width=self.net_w)(o) for o in outputs]
        merge = Concatenate()(outputs)
        output = self.multiple_conv2d(merge)
        output = self.convolutional_layer(filters=self.n_class, kernel_size=(1, 1), activation="softmax")(output)
        ensemble_name = "_".join([m.name for m in self.submodels])
        model = Model(inputs=inputs, outputs=output, name=ensemble_name)
        return model
