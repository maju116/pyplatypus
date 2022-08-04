from tensorflow.keras.layers import (
    SeparableConv2D, BatchNormalization, MaxPool2D, Dropout, Conv2DTranspose,
    Concatenate, Cropping2D, Resizing, Average, Add, Conv2D, SpatialDropout2D,
    UpSampling2D, ReLU, LeakyReLu, ZeroPadding2D
)
from tensorflow.keras import activations as KRACT
from tensorflow.keras.backend import int_shape
from tensorflow.keras import Model, Input
from tensorflow.keras.regularizers import L2
import tensorflow as tf
from typing import Tuple, Any, Optional, List

coco_anchors = [
    [(116 / 416, 90 / 416), (156 / 416, 198 / 416), (373 / 416, 326 / 416)],
    [(30 / 416, 61 / 416), (62 / 416, 45 / 416), (59 / 416, 119 / 416)],
    [(10 / 416, 13 / 416), (16 / 416, 30 / 416), (33 / 416, 23 / 416)]
]


class yolo3:

    def __init__(
            self,
            net_h: int,
            net_w: int,
            grayscale: bool,
            n_class: int = 80,
            anchors: List[List[Tuple, Tuple]] = coco_anchors,
            **kwargs
    ) -> None:
        """

        Args:
            net_h (int): Input layer height. Must be divisible by `32`.
            net_w (int): Input layer width. Must be divisible by `32`.
            grayscale (int): Defines input layer color channels -  `1` if `TRUE`, `3` if `FALSE`.
            n_class (int): Number of prediction classes.
            anchors (List[List[Tuple, Tuple]]): Prediction anchors.
        """
        self.net_h = net_h
        self.net_w = net_w
        self.grayscale = grayscale
        self.n_class = n_class
        self.anchors = anchors

    @staticmethod
    def darknet53_conv2d(
            input: tf.Tensor,
            strides: int,
            filters: int,
            kernel_size: int,
            batch_normalization: bool = True,
            leaky_relu: bool = True
    ) -> tf.Tensor:
        """
        Creates a convolutional Darknet53 unit.

        Args:
            input (tf.Tensor): Model or layer object.
            strides (int): An integer or list of 2 integers, specifying the strides of the convolution along the width and height.
            filters (int): Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
            kernel_size (int): An integer or list of 2 integers, specifying the width and height of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
            batch_normalization (bool): Should batch normalization be used in the unit.
            leaky_relu (bool): Should leaky ReLU activation function be used in the unit.

        Returns:
            Convolutional Darknet53 unit.
        """
        output = input
        if strides > 1:
            output = ZeroPadding2D(padding=((1, 0), (1, 0)))(output)
        output = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                        padding="valid" if strides > 1 else "same",
                        use_bias=False if batch_normalization else True,
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=L2(l2=5e-4))(output)
        if batch_normalization:
            output = BatchNormalization(center=True, scale=True, momentum=0.99, epsilon=1e-3)(output)
        if leaky_relu:
            output = LeakyReLu(alpha=0.1)
        return output

    def darknet53_residual_block(
            self,
            input: tf.Tensor,
            filters: int,
            blocks: int
    ) -> tf.Tensor:
        """
        Creates a residual Darknet53 block.

        Args:
            input (tf.Tensor): Model or layer object.
            filters (int): Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
            blocks (int): Number of residual blocks.

        Returns:
            Residual Darknet53 block.
        """
        output = self.darknet53_conv2d(input, strides=2, filters=filters, kernel_size=3,
                                       batch_normalization=True, leaky_relu=True)
        for block in range(blocks):
            add_layer = output
            output = self.darknet53_conv2d(output, strides=1, filters=round(filters / 2), kernel_size=1,
                                           batch_normalization=True, leaky_relu=True)
            output = self.darknet53_conv2d(output, strides=1, filters=filters, kernel_size=3,
                                           batch_normalization=True, leaky_relu=True)
            output = Add()([output, add_layer])
        return output

    def darknet53(
            self,
            channels: int
    ) -> tf.Tensor:
        """
        Creates a Darknet53 architecture.

        Args:
            channels (int): Number of channels.

        Returns:
            Darknet53 model.
        """
        input = Input(shape=(None, None, channels))
        output = self.darknet53_conv2d(input, strides=1, filters=32, kernel_size=3,
                                       batch_normalization=True, leaky_relu=True)
        output = self.darknet53_residual_block(output, filters=64, blocks=1)
        output = self.darknet53_residual_block(output, filters=128, blocks=2)
        output = self.darknet53_residual_block(output, filters=256, blocks=8)
        output1 = output
        output = self.darknet53_residual_block(output, filters=512, blocks=8)
        output2 = output
        output3 = self.darknet53_residual_block(output, filters=1024, blocks=4)
        return Model(inputs=input, outputs=[output1, output2, output3], name='darknet53')

    def yolo3_conv2d(
            self
    ) -> tf.Tensor:
        return 33
