from tensorflow.keras.layers import (
    SeparableConv2D, BatchNormalization, MaxPool2D, Dropout, Conv2DTranspose,
    Concatenate, Cropping2D, Resizing, Average, Add, Conv2D, SpatialDropout2D,
    UpSampling2D
    )
from tensorflow.keras import activations as KRACT
from tensorflow.keras.backend import int_shape
from tensorflow.keras import Model, Input
import tensorflow as tf
from typing import Tuple, List, Any, Dict


class u_net:

    def __init__(
            self,
            net_h: int,
            net_w: int,
            grayscale: bool,
            blocks: int = 4,
            n_class: int = 2,
            filters: int = 16,
            dropout: float = 0.1,
            batch_normalization: bool = True,
            kernel_initializer: str = "he_normal",
            linknet: bool = False,
            plus_plus: bool = False,
            deep_supervision: bool = False,
            use_separable_conv2d: bool = True,
            use_spatial_dropout2d: bool = True,
            use_up_sampling2d: bool = False,
            activation_function_name: str = "ReLU",
            **kwargs
    ) -> None:
        """
        Creates U-Net model architecture.

        Args:
            net_h (int): Input layer height. Must be equal to `2^x, x - natural`.
            net_w (int): Input layer width. Must be equal to `2^x, x - natural`.
            grayscale (bool): Defines input layer color channels -  `1` if `True`, `3` if `False`.
            blocks (int): Number of blocks in the model.
            n_class (int): Number of classes. Minimum is `2` (background + other object).
            filters (int): Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
            dropout (float): Dropout rate.
            batch_normalization (bool): Should batch normalization be used in the block.
            kernel_initializer (str): Initializer for the kernel weights matrix.
            linknet (bool): Should Linknet connections (Add) instead of U-Net connections (Concatenate) be used.
            plus_plus (bool): Should U-Net++ instead od U-Net architecture be used.
            deep_supervision (bool): Should deep supervision be used when using U-Net++ architecture.
        """
        self.type = 'u_net'
        self.net_h = net_h
        self.net_w = net_w
        self.grayscale = grayscale
        self.blocks = blocks
        self.n_class = n_class
        self.filters = filters
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.kernel_initializer = kernel_initializer
        self.linknet = linknet
        self.plus_plus = plus_plus
        self.deep_supervision = deep_supervision
        self.use_separable_conv2d = use_separable_conv2d
        self.use_spatial_droput2d = use_spatial_dropout2d
        self.use_up_sampling2d = use_up_sampling2d
        self.activation_function_name = activation_function_name
        self.model = self.build_model()

    def dropout_layer(self):
        if self.use_spatial_droput2d:
            droput_layer = SpatialDropout2D(rate=self.dropout)
        else:
            droput_layer = Dropout(rate=self.dropout)
        return droput_layer

    def u_net_multiple_conv2d(
            self,
            input: tf.Tensor,
            filters: int,
            kernel_size: Tuple[int, int],
            u_net_conv_block_width: int = 2
    ) -> tf.Tensor:
        """
        Creates a double convolutional U-Net block.

        Args:
            input (tf.Tensor): Model or layer object.
            filters (int): Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
            kernel_size (Tuple[int, int]): An integer or tuple of 2 integers, specifying the width and height of the 2D convolution window. 
            Can be a single integer to specify the same value for all spatial dimensions.
            u_net_conv_block_width (int): Controls the amount of convolutional layers in the block.

        Returns:
            Double convolutional bloc of U-Net model.
        """
        for i in range(u_net_conv_block_width):
            if self.use_separable_conv2d:
                input = SeparableConv2D(
                    filters=filters, kernel_size=kernel_size, padding="same",
                    kernel_initializer=self.kernel_initializer)(
                        input
                    )
            else:
                input = Conv2D(
                    filters=filters, kernel_size=kernel_size, padding="same",
                    kernel_initializer=self.kernel_initializer)(
                        input
                    )
            if self.batch_normalization:
                input = BatchNormalization()(input)
            input = self.activation_layer()(input)
        return input

    def activation_layer(self):
        activation_layer = getattr(KRACT, self.activation_function_name)
        return activation_layer

    @staticmethod
    def get_crop_shape(target, reference):
        height_change = target[1] - reference[1]
        assert (height_change >= 0)
        if height_change % 2 != 0:
            ch1, ch2 = int(height_change / 2), int(height_change / 2) + 1
        else:
            ch1, ch2 = int(height_change / 2), int(height_change / 2)
        width_change = target[2] - reference[2]
        assert (width_change >= 0)
        if width_change % 2 != 0:
            cw1, cw2 = int(width_change / 2), int(width_change / 2) + 1
        else:
            cw1, cw2 = int(width_change / 2), int(width_change / 2)

        return (ch1, ch2), (cw1, cw2)

    def init_empty_layers_placeholders(
            self
    ) -> tuple[list[Any], list[Any], dict[Any, Any]]:
        """
        Creates layers placeholders.

        Returns:
            Layers placeholders.
        """
        conv_layers = []
        pool_layers = []
        subconv_layers = {}
        if self.plus_plus:
            for block in range(self.blocks):
                subconv_layers[block] = []
        return conv_layers, pool_layers, subconv_layers

    def generate_input(
            self
    ) -> tf.Tensor:
        """
        Generates input for U-Net/U-Net++ model.

        Returns:
            Input for U-Net/U-Net++ model.
        """
        channels = 1 if self.grayscale else 3
        input_shape = (self.net_h, self.net_w, channels)
        return Input(shape=input_shape, name='input_img')

    def generate_output(
            self,
            output_tensor: tf.Tensor,
            subconv_layers: dict
    ) -> tf.Tensor:
        """
        Generates output for U-Net/U-Net++ model.

        Args:
            output_tensor (tf.Tensor): Output tensor.
            subconv_layers (list): Sub-convolutional layers of U-Net++.

        Returns:
            Output for U-Net/U-Net++ model.
        """
        conv_layer = SeparableConv2D if self.use_separable_conv2d else Conv2D
        output = conv_layer(self.n_class, 1, activation="softmax", padding="same")(output_tensor)
        output = Resizing(height=self.net_h, width=self.net_w)(output)
        if self.plus_plus and self.deep_supervision:
            outputs = subconv_layers[0].copy()
            outputs = [conv_layer(self.n_class, 1, activation="softmax", padding="same")(o) for o in outputs]
            outputs = [Resizing(height=self.net_h, width=self.net_w)(o) for o in outputs]
            outputs.append(output)
            output = Average()(outputs)
        return output

    def horizontal_connection(
            self
    ):
        """
        Generates function for horizontal connection.

        Returns:
            Horizontal connection function.
        """
        return Add if self.linknet else Concatenate

    def build_model(
            self
    ) -> tf.keras.Model:
        """
        Creates a U-Net architecture.

        Returns:
            U-Net/U-Net++ model.
        """
        input_img = self.generate_input()
        conv_layers, pool_layers, subconv_layers = self.init_empty_layers_placeholders()
        for block in range(self.blocks):
            current_input = input_img if block == 0 else pool_layers[block - 1]
            current_input = self.u_net_multiple_conv2d(current_input, self.filters * 2 ** block, kernel_size=(3, 3))
            conv_layers.append(current_input)
            current_input = MaxPool2D(pool_size=2)(current_input)
            current_input = self.dropout_layer()(current_input)
            pool_layers.append(current_input)
            if self.plus_plus:
                for subblock in list(reversed(range(block))):
                    if subblock == block - 1:
                        down_layer = conv_layers[block]
                    else:
                        down_layer = subconv_layers[subblock + 1][-1]
                    if not self.use_up_sampling2d:
                        down_layer = Conv2DTranspose(
                            self.filters * 2 ** (self.blocks - block - 1),
                            kernel_size=(3, 3), strides=2, padding="same"
                            )(down_layer)
                    else:
                        down_layer = UpSampling2D((2, 2))(down_layer)
                        down_layer = Conv2D(
                            self.filters * 2 ** (self.blocks - block - 1), kernel_size=(3, 3), strides=(2, 2), padding="same"
                        )(down_layer)
                    left_layers = subconv_layers[subblock].copy()
                    left_layers.append(down_layer)
                    ch = int_shape(conv_layers[subblock])[1]
                    cw = int_shape(conv_layers[subblock])[2]
                    left_layers = [Resizing(height=ch, width=cw)(lr) for lr in left_layers]
                    left_layers.append(conv_layers[subblock])
                    subblock_layer = Concatenate()(left_layers)
                    subblock_layer = self.u_net_multiple_conv2d(subblock_layer, self.filters * 2 ** block,
                                                              kernel_size=(3, 3))
                    subconv_layers[subblock].append(subblock_layer)
        current_input = self.u_net_multiple_conv2d(current_input, self.filters * 2 ** self.blocks, kernel_size=(3, 3))
        conv_layers.append(current_input)
        for block in range(self.blocks):
            if not self.use_up_sampling2d:
                current_input = Conv2DTranspose(self.filters * 2 ** (self.blocks - block - 1),
                                                kernel_size=(3, 3), strides=2, padding="same")(
                    conv_layers[self.blocks + block])
            else:
                current_input = UpSampling2D((2, 2))(conv_layers[self.blocks + block])

            ch, cw = self.get_crop_shape(int_shape(conv_layers[self.blocks - block - 1]), int_shape(current_input))
            if self.plus_plus:
                current_input = Concatenate()([current_input,
                                               Cropping2D(cropping=(ch, cw))(conv_layers[self.blocks - block - 1]),
                                               ] + [Cropping2D(cropping=(ch, cw))(lr) for lr in
                                                    subconv_layers[self.blocks - block - 1]])
            else:
                current_input = self.horizontal_connection()()([current_input,
                                                                Cropping2D(cropping=(ch, cw))(
                                                                    conv_layers[self.blocks - block - 1]),
                                                                ])
            current_input = self.dropout_layer()(current_input)
            current_input = self.u_net_multiple_conv2d(current_input, self.filters * 2 ** (self.blocks - block - 1),
                                                     kernel_size=(3, 3))
            conv_layers.append(current_input)
        output = self.generate_output(conv_layers[2 * self.blocks], subconv_layers)
        return Model(inputs=input_img, outputs=output, name="u_net")
