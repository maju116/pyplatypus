from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, Dropout, Conv2DTranspose, Concatenate
from tensorflow.keras import Model, Input
from typing import Tuple, List, Optional


def u_net_double_conv2d(
        input,
        filters: int,
        kernel_size: Tuple[int, int],
        batch_normalization: bool,
        kernel_initializer: str
):
    """
    Creates a double convolutional U-Net block.

    Args:
     input (): Model or layer object.
     filters (int): Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
     kernel_size (Tuple[int, int]): An integer or tuple of 2 integers, specifying the width and height of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
     batch_normalization (bool): Should batch normalization be used in the block.
     kernel_initializer (str): Initializer for the kernel weights matrix.
    """
    for i in range(2):
        input = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", kernel_initializer=kernel_initializer)(
            input)
        if batch_normalization:
            input = BatchNormalization()(input)
        input = ReLU()(input)
    return input


def u_net(
        net_h: int,
        net_w: int,
        grayscale: bool,
        blocks: int = 4,
        n_class: int = 2,
        filters: int = 16,
        dropout: float = 0.1,
        batch_normalization: bool = True,
        kernel_initializer: str = "he_normal"):
    """
    Creates a U-Net architecture.

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
    # Add checks
    channels = 1 if grayscale else 3
    input_shape = (net_h, net_w, channels)
    input_img = Input(shape=input_shape, name='input_img')
    conv_layers = []
    pool_layers = []
    conv_tr_layers = []
    for block in range(blocks):
        current_input = input_img if block == 0 else pool_layers[block - 1]
        current_input = u_net_double_conv2d(current_input, filters * 2 ** block, kernel_size=(3, 3),
                                            batch_normalization=batch_normalization,
                                            kernel_initializer=kernel_initializer)
        conv_layers.append(current_input)
        current_input = MaxPool2D(pool_size=2)(current_input)
        current_input = Dropout(rate=dropout)(current_input)
        pool_layers.append(current_input)
    current_input = u_net_double_conv2d(current_input, filters * 2 ** blocks, kernel_size=(3, 3),
                                        batch_normalization=batch_normalization,
                                        kernel_initializer=kernel_initializer)
    conv_layers.append(current_input)
    for block in range(blocks):
        current_input = Conv2DTranspose(filters * 2 ** (blocks - block - 1),
                                        kernel_size=(3, 3), strides=2, padding="same")(conv_layers[blocks + block])
        current_input = Concatenate()([current_input, conv_layers[blocks - block - 1]])
        current_input = Dropout(rate=dropout)(current_input)
        conv_tr_layers.append(current_input)
        current_input = u_net_double_conv2d(current_input, filters * 2 ** (blocks - block - 1), kernel_size=(3, 3),
                                            batch_normalization=batch_normalization,
                                            kernel_initializer=kernel_initializer)
        conv_layers.append(current_input)
    output = Conv2D(n_class, 1, activation="softmax")(conv_layers[2 * blocks])
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