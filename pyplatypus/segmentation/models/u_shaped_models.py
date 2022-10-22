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


class u_shaped_model:

    def __init__(
        self,
        net_h: int,
        net_w: int,
        channels: Union[int, List[int]],
        blocks: Optional[int] = 4,
        n_class: Optional[int] = 2,
        filters: Optional[int] = 16,
        dropout: Optional[float] = 0.1,
        batch_normalization: Optional[bool] = True,
        kernel_initializer: Optional[str] = "he_normal",
        resunet: Optional[bool] = False,
        linknet: Optional[bool] = False,
        plus_plus: Optional[bool] = False,
        deep_supervision: Optional[bool] = False,
        use_separable_conv2d: Optional[bool] = True,
        use_spatial_dropout2d: Optional[bool] = True,
        use_up_sampling2d: Optional[bool] = False,
        activation_layer: Optional[str] = "relu",
        **kwargs
            ) -> None:
        """Creates U-Net model architecture.

        Parameters
        ----------
        net_h : int
            Input layer height.
        net_w : int
            Input layer width.
        channels : int
            Defines input layer color channels.
        blocks : Optional[int], optional
            Number of blocks in the model, by default 4
        n_class : Optional[int], optional
            Number of classes. Minimum is `2` (background + other object), by default 2
        filters : Optional[int], optional
            Integer, dimensionality of the output space (i.e. the number of output filters in the convolution), by default 16
        dropout : Optional[float], optional
            Dropout rate, by default 0.1
        batch_normalization : Optional[bool], optional
            Should batch normalization be used in the block., by default True
        kernel_initializer : Optional[str], optional
            Initializer for the kernel weights matrix, by default "he_normal"
        resunet : Optional[bool], optional
            Should Res-U-Net connections (Residual) instead of U-Net connections (Concatenate) be used, by default False
        linknet : Optional[bool], optional
            Should Linknet connections (Add) instead of U-Net connections (Concatenate) be used, by default False
        plus_plus : Optional[bool], optional
            Should U-Net++ instead od U-Net architecture be used, by default False
        deep_supervision : Optional[bool], optional
            Should deep supervision be used when using U-Net++ architecture, by default False
        use_separable_conv2d : Optional[bool], optional
             Determines if the SeparableConv2D layers should be used, if set to false, the Conv2D is used, by default True
        use_spatial_dropout2d : Optional[bool], optional
            Indicates whether the spatial or regular droput should be used, by default True
        use_up_sampling2d : Optional[bool], optional
            If set to False, the transpozed convolutional layer is used, by default False
        activation_layer : Optional[str], optional
            Allows the user to choose any activation layer available in the tensorflow.keras.activations, by default "relu"
        """
        self.net_h = net_h
        self.net_w = net_w
        self.channels = sum(channels) if isinstance(channels, list) else channels
        self.blocks = blocks
        self.n_class = n_class
        self.filters = filters
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.kernel_initializer = kernel_initializer
        self.resunet = resunet
        self.linknet = linknet
        self.plus_plus = plus_plus
        self.deep_supervision = deep_supervision
        self.use_separable_conv2d = use_separable_conv2d
        self.use_spatial_droput2d = use_spatial_dropout2d
        self.use_up_sampling2d = use_up_sampling2d
        self.activation_layer = activation_layer
        self.model = self.build_model()

    def dropout_layer(self):
        """Creates the dropout layer of the preferred kind.

        Returns
        -------
        dropout_layer: KerasTensor
            Dropout created by the chosen method.
        """
        if self.use_spatial_droput2d:
            dropout_layer = SpatialDropout2D(rate=self.dropout)
        else:
            dropout_layer = Dropout(rate=self.dropout)
        return dropout_layer

    def convolutional_layer(
        self, filters: int, kernel_size: Tuple[int, int], activation: Optional[str] = "relu"
            ) -> Union[SeparableConv2D, Conv2D]:
        """
        Returns the convolutional layer of the demanded type.

        Parameters
        ----------
        input: tf.Tensor
            Model or layer object.
        filters: int
            Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
        kernel_size: Tuple[int, int])
            An integer or tuple of 2 integers, specifying the width and height of the 2D convolution window.
            If single integer is supplied the same value is used for all spatial dimensions.

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

    def u_net_multiple_conv2d(
            self,
            input: tf.Tensor,
            filters: int,
            kernel_size: Tuple[int, int],
            u_net_conv_block_width: int = 2
            ) -> tf.Tensor:
        """
        Creates a multiple convolutional U-Net block.

        Parameters
        ----------
        input: tf.Tensor
            Model or layer object.
        filters: int
            Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
        kernel_size: Tuple[int, int]
            An integer or tuple of 2 integers, specifying the width and height of the 2D convolution window.
            It is allowed to be a single integer to specify the same value for all spatial dimensions.
        u_net_conv_block_width: int
            Controls the amount of convolutional layers in the block, by default 2

        Returns
        -------
        input:
            Multiple convolutional block of the U-Net model.
        """
        for i in range(u_net_conv_block_width):
            input = self.convolutional_layer(filters, kernel_size)(input)
            if self.batch_normalization:
                input = BatchNormalization()(input)
            input = self.activation()(input)
        return input

    def res_u_net_multiple_conv2d(
        self,
        input: tf.Tensor,
        filters: int,
        kernel_size: Tuple[int, int],
        res_u_net_conv_block_width: int = 2
            ) -> tf.Tensor:
        """
        Creates a multiple convolutional Res-U-Net block, with the raw input added before the final activation.

        Parameters
        ----------
        input: tf.Tensor
            Model or layer object.
        filters: int
            Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
        kernel_size: Tuple[int, int]
            An integer or tuple of 2 integers, specifying the width and height of the 2D convolution window.
            It is allowed to be a single integer to specify the same value for all spatial dimensions.
        res_u_net_conv_block_width: int
            Controls the amount of convolutional layers in the block, by default 2.

        Returns:
            Multiple convolutional bloc of U-Net model.
        """
        # TODO The input number of filters must be the same as the one used within the block, to discuss.
        raw_input = Conv2D(
            filters=filters, kernel_size=kernel_size, padding="same",
            kernel_initializer=self.kernel_initializer)(
                input
            )
        for i in range(res_u_net_conv_block_width):
            input = self.convolutional_layer(filters, kernel_size)(input)
            if self.batch_normalization:
                input = BatchNormalization()(input)
        # Add the input to the block output and let it flow through the ReLU and BN.
        input = Add()([raw_input, input])
        input = self.activation()(input)
        if self.batch_normalization:
            input = BatchNormalization()(input)
        return input

    def multiple_conv2d_block(
        self,
        input: tf.Tensor,
        filters: int,
        kernel_size: Tuple[int, int],
        conv_block_width: int = 2
            ) -> tf.Tensor:
        """
        Creates a multiple convolutional block, suiting the chosen architecture.

        Parameters
        ----------
        input: tf.Tensor
            Model or layer object.
        filters: int
            Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
        kernel_size: Tuple[int, int]
            An integer or tuple of 2 integers, specifying the width and height of the 2D convolution window.
            It is allowed to be a single integer to specify the same value for all spatial dimensions.
        conv_block_width: int
            Controls the amount of convolutional layers in the block, by default 2.

        Returns
        -------
        conv_block:
            Multiple convolutional block of the model.
        """
        if self.resunet:
            conv_block = self.res_u_net_multiple_conv2d(input, filters, kernel_size, conv_block_width)
        else:
            conv_block = self.u_net_multiple_conv2d(input, filters, kernel_size, conv_block_width)
        return conv_block

    def activation(self):
        """Creates the layer applying the specified activation function.

        Returns
        -------
        activation_layer: function
            Layer later used to apply the chosen activation function.
        """
        activation_layer = getattr(KRACT, self.activation_layer)
        return activation_layer

    @staticmethod
    def get_crop_shape(target: tuple, reference: tuple) -> Tuple[tuple]:
        """Creates the cropping specifications exemplary used in the Cropping2D layer.
        Based on the reference shape, it decides how many rows/columns should be dropped from each side
        of the input frame.

        Parameters
        ----------
        target: tuple
            The shape of frame to be cropped.
        reference: tuple
            The desired shape.

        Returns
        -------
        ch: tuple
            Vertical crop specification.
        cw: tuple
            Horizontal crop specification.
        """
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
        ch, cw = (ch1, ch2), (cw1, cw2)
        return ch, cw

    def init_empty_layers_placeholders(self) -> tuple[list[Any], list[Any], dict[Any, Any]]:
        """
        Creates layers placeholders.

        Returns
        -------
        Layers placeholders.
        """
        conv_layers = []
        pool_layers = []
        subconv_layers = {}
        if self.plus_plus:
            for block in range(self.blocks):
                subconv_layers[block] = []
        return conv_layers, pool_layers, subconv_layers

    def generate_input(self) -> tf.Tensor:
        """
        Generates input for U-Net/U-Net++ model.

        Returns
        -------
        input_layer:
            Input for U-Net/U-Net++ model.
        """
        input_shape = (self.net_h, self.net_w, self.channels)
        input_layer = Input(shape=input_shape, name='input_img')
        return input_layer

    def generate_output(self, output_tensor: tf.Tensor, subconv_layers: dict) -> tf.Tensor:
        """
        Generates output for U-Net/U-Net++ model.

        Parameters
        ----------
        output_tensor: tf.Tensor
            Output tensor.
        subconv_layers: list
            Sub-convolutional layers of U-Net++.

        Returns
        -------
        output:
            Output for U-Net/U-Net++ model.
        """
        output = self.convolutional_layer(filters=self.n_class, kernel_size=1, activation="softmax")(output_tensor)
        output = Resizing(height=self.net_h, width=self.net_w)(output)
        if self.plus_plus and self.deep_supervision:
            outputs = subconv_layers[0].copy()
            outputs = [self.convolutional_layer(filters=self.n_class, kernel_size=1, activation="softmax")(o) for o in outputs]
            outputs = [Resizing(height=self.net_h, width=self.net_w)(o) for o in outputs]
            outputs.append(output)
            output = Average()(outputs)
        return output

    def horizontal_connection(self):
        """
        Generates function for horizontal connection.

        Returns
        -------
        horizontal_connection:
            Horizontal connection function.
        """
        horizontal_connection = Add if self.linknet else Concatenate
        return horizontal_connection

    def build_model(self) -> tf.keras.Model:
        """
        Creates a U-Net architecture.

        Returns
        -------
        model:
            U-Net/U-Net++ model.
        """
        input_img = self.generate_input()
        conv_layers, pool_layers, subconv_layers = self.init_empty_layers_placeholders()
        for block in range(self.blocks):
            current_input = input_img if block == 0 else pool_layers[block - 1]
            current_input = self.multiple_conv2d_block(
                input=current_input, filters=self.filters * 2 ** block, kernel_size=(3, 3)
                )
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
                    subblock_layer = self.multiple_conv2d_block(
                        input=subblock_layer, filters=self.filters * 2 ** block, kernel_size=(3, 3)
                        )
                    subconv_layers[subblock].append(subblock_layer)
        current_input = self.multiple_conv2d_block(
            input=current_input, filters=self.filters * 2 ** self.blocks, kernel_size=(3, 3)
            )
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
                current_input = self.horizontal_connection()()([
                    current_input, Cropping2D(cropping=(ch, cw))(conv_layers[self.blocks - block - 1]),
                    ])
            current_input = self.dropout_layer()(current_input)
            current_input = self.multiple_conv2d_block(
                input=current_input, filters=self.filters * 2 ** (self.blocks - block - 1), kernel_size=(3, 3)
                )
            conv_layers.append(current_input)
        output = self.generate_output(conv_layers[2 * self.blocks], subconv_layers)
        model = Model(inputs=input_img, outputs=output, name="u_net")
        return model
