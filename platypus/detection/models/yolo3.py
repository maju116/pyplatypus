from tensorflow.keras.layers import (
    SeparableConv2D, BatchNormalization, MaxPool2D, Dropout, Conv2DTranspose,
    Concatenate, Cropping2D, Resizing, Average, Add, Conv2D, SpatialDropout2D,
    UpSampling2D, ReLU, LeakyReLU, ZeroPadding2D, Reshape
)
from tensorflow.keras import Model, Input
from tensorflow.keras.regularizers import L2
import tensorflow as tf
from typing import Tuple, List, Union

coco_anchors = [
    [(116 / 416, 90 / 416), (156 / 416, 198 / 416), (373 / 416, 326 / 416)],
    [(30 / 416, 61 / 416), (62 / 416, 45 / 416), (59 / 416, 119 / 416)],
    [(10 / 416, 13 / 416), (16 / 416, 30 / 416), (33 / 416, 23 / 416)]
]

coco_labels = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")


class yolo3:

    def __init__(
            self,
            net_h: int,
            net_w: int,
            grayscale: bool,
            n_class: int = 80,
            anchors: List[List[Tuple]] = coco_anchors,
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
            output = LeakyReLU(alpha=0.1)(output)
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
            self,
            inputs: Union[tf.Tensor, List[tf.Tensor]],
            filters: int,
            name: str
    ) -> tf.Tensor:
        """
        Creates a convolutional Yolo3 unit.

        Args:
            inputs (Union[tf.Tensor, List[tf.Tensor]]): Models or layer objects.
            filters (int): Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
            name (str): Model name.

        Returns:
            Convolutional Yolo3 unit.
        """
        if type(inputs) == list:
            input1 = Input(shape=inputs[0].shape.as_list()[1:4])
            input2 = Input(shape=inputs[1].shape.as_list()[1:4])
            input = [input1, input2]
            net_out = self.darknet53_conv2d(input1, strides=1, filters=filters,
                                            kernel_size=1, batch_normalization=True,
                                            leaky_relu=True)
            net_out = UpSampling2D(size=2)(net_out)
            net_out = Concatenate()([net_out, input2])
        else:
            input = Input(shape=inputs.shape.as_list()[1:4])
            net_out = input
        net_out = self.darknet53_conv2d(net_out, strides=1, filters=filters,
                                        kernel_size=1, batch_normalization=True,
                                        leaky_relu=True)
        net_out = self.darknet53_conv2d(net_out, strides=1, filters=filters * 2,
                                        kernel_size=3, batch_normalization=True,
                                        leaky_relu=True)
        net_out = self.darknet53_conv2d(net_out, strides=1, filters=filters,
                                        kernel_size=1, batch_normalization=True,
                                        leaky_relu=True)
        net_out = self.darknet53_conv2d(net_out, strides=1, filters=filters * 2,
                                        kernel_size=3, batch_normalization=True,
                                        leaky_relu=True)
        net_out = self.darknet53_conv2d(net_out, strides=1, filters=filters,
                                        kernel_size=1, batch_normalization=True,
                                        leaky_relu=True)
        return Model(input, net_out, name=name)(inputs)

    def yolo3_output(
            self,
            inputs: Union[tf.Tensor, List[tf.Tensor]],
            filters: int,
            anchors_per_grid: int,
            n_class: int,
            name: str
    ) -> tf.Tensor:
        """
        Creates Yolo3 output grid of dimensionality `(S, H, W, anchors_per_grid, 5 + n_class)`.

        Args:
            inputs (Union[tf.Tensor, List[tf.Tensor]]): Models or layer objects.
            filters (int): Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
            anchors_per_grid (int): Number of anchors/boxes per one output grid.
            n_class (int): Number of prediction classes.
            name (str): Model name.

        Returns:
            Yolo3 output grid.
        """
        input = Input(shape=inputs.shape.as_list()[1:4])
        net_out = self.darknet53_conv2d(input, strides=1, filters=filters * 2,
                                        kernel_size=3, batch_normalization=True,
                                        leaky_relu=True)
        net_out = self.darknet53_conv2d(net_out, strides=1, filters=anchors_per_grid * (n_class + 5),
                                        kernel_size=1, batch_normalization=False,
                                        leaky_relu=False)
        net_out_shape = net_out.shape.as_list()
        net_out = Reshape(target_shape=(net_out_shape[1], net_out_shape[2],
                                        anchors_per_grid, n_class + 5))(net_out)
        return Model(input, net_out, name=name)(inputs)

    def yolo3(
            self
    ) -> tf.Tensor:
        """
        Creates a Yolo3 architecture.

        Returns:
            Yolo3 model.
        """
        anchors_per_grid = len(self.anchors[0])
        channels = 1 if self.grayscale else 3
        input_shape = (self.net_h, self.net_w, channels)
        input_img = Input(shape=input_shape, name='input_img')
        darknet = self.darknet53(channels)(input_img)
        net_out = self.yolo3_conv2d(darknet[2], 512, name="yolo3_conv1")
        grid_1 = self.yolo3_output(net_out, 512, anchors_per_grid, self.n_class, name="grid1")
        net_out = self.yolo3_conv2d([net_out, darknet[1]], 256, name="yolo3_conv2")
        grid_2 = self.yolo3_output(net_out, 256, anchors_per_grid, self.n_class, name="grid2")
        net_out = self.yolo3_conv2d([net_out, darknet[0]], 128, name="yolo3_conv3")
        grid_3 = self.yolo3_output(net_out, 128, anchors_per_grid, self.n_class, name="grid3")
        return Model(input_img, (grid_1, grid_2, grid_3), name="yolo3")
