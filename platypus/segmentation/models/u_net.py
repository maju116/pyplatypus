from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, ReLU, MaxPool2D, Dropout, Conv2DTranspose, \
    Concatenate, Cropping2D, Resizing
from tensorflow.keras.backend import int_shape
from tensorflow.keras import Model, Input
import tensorflow as tf
from typing import Tuple


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
            kernel_initializer: str = "he_normal"
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
        self.model = self.build_model()

    def u_net_double_conv2d(
            self,
            input: tf.Tensor,
            filters: int,
            kernel_size: Tuple[int, int]
    ) -> tf.Tensor:
        """
        Creates a double convolutional U-Net block.

        Args:
         input (tf.Tensor): Model or layer object.
         filters (int): Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
         kernel_size (Tuple[int, int]): An integer or tuple of 2 integers, specifying the width and height of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.

        Returns:
            Double convolutional bloc of U-Net model.
        """
        for i in range(2):
            input = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding="same",
                                    kernel_initializer=self.kernel_initializer)(
                input)
            if self.batch_normalization:
                input = BatchNormalization()(input)
            input = ReLU()(input)
        return input

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

    def build_model(
            self
    ) -> tf.keras.Model:
        """
        Creates a U-Net architecture.

        Returns:
            U-Net model.
        """
        channels = 1 if self.grayscale else 3
        input_shape = (self.net_h, self.net_w, channels)
        input_img = Input(shape=input_shape, name='input_img')
        conv_layers = []
        pool_layers = []
        for block in range(self.blocks):
            current_input = input_img if block == 0 else pool_layers[block - 1]
            current_input = self.u_net_double_conv2d(current_input, self.filters * 2 ** block, kernel_size=(3, 3))
            conv_layers.append(current_input)
            current_input = MaxPool2D(pool_size=2)(current_input)
            current_input = Dropout(rate=self.dropout)(current_input)
            pool_layers.append(current_input)
        current_input = self.u_net_double_conv2d(current_input, self.filters * 2 ** self.blocks, kernel_size=(3, 3))
        conv_layers.append(current_input)
        for block in range(self.blocks):
            current_input = Conv2DTranspose(self.filters * 2 ** (self.blocks - block - 1),
                                            kernel_size=(3, 3), strides=2, padding="same")(
                conv_layers[self.blocks + block])
            ch, cw = self.get_crop_shape(int_shape(conv_layers[self.blocks - block - 1]), int_shape(current_input))
            current_input = Concatenate()([current_input,
                                           Cropping2D(cropping=(ch, cw))(conv_layers[self.blocks - block - 1]),
                                           ])
            current_input = Dropout(rate=self.dropout)(current_input)
            current_input = self.u_net_double_conv2d(current_input, self.filters * 2 ** (self.blocks - block - 1),
                                                     kernel_size=(3, 3))
            conv_layers.append(current_input)
        output = SeparableConv2D(self.n_class, 1, activation="softmax", padding="same")(conv_layers[2 * self.blocks])
        output = Resizing(height=self.net_h, width=self.net_w)(output)
        return Model(inputs=input_img, outputs=output, name="u_net")


voc_labels = ('background', 'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person',
              'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor')

voc_colormap = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                (0, 64, 128)]

binary_colormap = [(0, 0, 0), (255, 255, 255)]

binary_labels = ('background', 'object')
