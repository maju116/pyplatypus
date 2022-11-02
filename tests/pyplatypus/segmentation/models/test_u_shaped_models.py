from pyplatypus.segmentation.models.u_shaped_models import u_shaped_model
from pyplatypus.data_models.semantic_segmentation import SemanticSegmentationModelSpec

from tensorflow.keras.layers import (
    SeparableConv2D, BatchNormalization, MaxPool2D, Dropout, Conv2DTranspose,
    Concatenate, Cropping2D, Resizing, Average, Add, Conv2D, SpatialDropout2D,
    UpSampling2D, ReLU
    )

from tensorflow.keras import Input
from numpy import array
import pytest
from tensorflow import math as tf_math


class TestUShapedModel:
    model_cfg = SemanticSegmentationModelSpec(**{
        "name": "model_name",
        "net_h": 8,
        "net_w": 8,
        "channels": 1,
        "blocks": 4,
        "n_class": 2,
        "filters": 2,
        "dropout": .1,
        "batch_normalization": True,
        "kernel_initializer": "he_normal",
        "resunet": False,
        "linknet": False,
        "plus_plus": False,
        "deep_supervision": False,
        "use_separable_conv2d": True,
        "use_spatial_dropout2d":True,
        "use_up_sampling2d": False,
        "activation_layer": "relu",
        "u_net_conv_block_width": 1
        })

    u_shaped_path = "pyplatypus.segmentation.models.u_shaped_models.u_shaped_model"
    raw_input = Conv2D(filters=2, kernel_size=(3, 3), padding="same")(Input(shape=(8, 8, 3), name='input_img'))

    @staticmethod
    def mocked_activation(layer):
        return layer

    def initialize_model(self, mocker):
        mocker.patch(self.u_shaped_path + ".build_model", return_value="built_model")
        return u_shaped_model(**dict(self.model_cfg))

    def test_init(self, mocker):
        mocker.patch(self.u_shaped_path + ".build_model", return_value="built_model")
        assert u_shaped_model(**dict(self.model_cfg)).model == "built_model"

    @pytest.mark.parametrize("use_spatial_dropout, expected_dropout", [(True, SpatialDropout2D), (False, Dropout)])
    def test_dropout_layer(self, mocker, use_spatial_dropout, expected_dropout):
        model = self.initialize_model(mocker)
        model.use_spatial_droput2d = use_spatial_dropout
        assert isinstance(model.dropout_layer(), expected_dropout)

    @pytest.mark.parametrize("use_separable_conv, expected_conv", [(True, SeparableConv2D), (False, Conv2D)])
    def test_conv_layer(self, mocker, use_separable_conv, expected_conv):
        model = self.initialize_model(mocker)
        model.use_separable_conv2d = use_separable_conv
        assert isinstance(model.convolutional_layer(filters=4, kernel_size=3), expected_conv)

    @pytest.mark.parametrize(
        "batch_normalization, expected_conv",
        [
            (False, raw_input),
            (True, BatchNormalization()(raw_input))
        ]
        )
    def test_multiple_conv_layer(self, mocker, batch_normalization, expected_conv):
        model = self.initialize_model(mocker)
        input_layer = Input(shape=(8, 8, 3), name='input_img')
        mocker.patch(self.u_shaped_path + ".activation", return_value=self.mocked_activation)
        mocker.patch(
            self.u_shaped_path + ".convolutional_layer",
            return_value=Conv2D(filters=2, kernel_size=(3, 3), padding="same")
            )
        model.batch_normalization = batch_normalization
        assert model.u_net_multiple_conv2d(
            input=input_layer,
            filters=2,
            kernel_size=(3, 3)
            ).shape[1:] == expected_conv.shape[1:]

    @pytest.mark.parametrize(
    "batch_normalization, expected_conv",
    [
        (False, Add()([raw_input, raw_input])),
        (True, BatchNormalization()(Add()([raw_input, BatchNormalization()(raw_input)])))
    ]
    )
    def test_multiple_res_u_net_layer(self, mocker, batch_normalization, expected_conv):
        model = self.initialize_model(mocker)
        input_layer = Input(shape=(8, 8, 3), name='input_img')
        mocker.patch(self.u_shaped_path + ".activation", return_value=self.mocked_activation)
        mocker.patch(self.u_shaped_path + ".convolutional_layer", return_value=self.mocked_activation)
        mocker.patch(
            self.u_shaped_path + ".convolutional_layer",
            return_value=Conv2D(filters=2, kernel_size=(3, 3), padding="same")
            )
        model.batch_normalization = batch_normalization
        assert model.res_u_net_multiple_conv2d(
            input=input_layer,
            filters=2,
            kernel_size=(3, 3)
            ).shape[1:] == expected_conv.shape[1:]
        model.res_u_net_multiple_conv2d(
            input=input_layer,
            filters=2,
            kernel_size=(3, 3)
            )

    @pytest.mark.parametrize("use_resunet, result", [(True, "res_u_net_block"), (False, "u_net_block")])
    def test_multiple_conv2d_block(self, mocker, use_resunet, result):
        mocker.patch(self.u_shaped_path + ".res_u_net_multiple_conv2d", return_value="res_u_net_block")
        mocker.patch(self.u_shaped_path + ".u_net_multiple_conv2d", return_value="u_net_block")
        model = self.initialize_model(mocker)
        model.resunet = use_resunet
        assert model.multiple_conv2d_block(input=Input((2, 2, 2)), filters=2, kernel_size=(3, 3)) == result


    def test_activation(self, monkeypatch, mocker):
        monkeypatch.setattr("pyplatypus.segmentation.models.u_shaped_models.KRACT.activation_1", "activation_function_1", raising=False)
        model = self.initialize_model(mocker)
        model.activation_layer = "activation_1"
        assert model.activation() == "activation_function_1"


    target = (20, 20)
    reference = (10, 10)

    target = (20, 20)
    reference = (12, 12)

    @pytest.mark.parametrize(
        "target, reference, add_factor",
        [
            ((1, 20, 20), (1, 10, 10), 0), 
            ((1, 20, 20), (1, 13, 13), 1)
        ]
        )
    def test_get_crop_shape(self, target, reference, add_factor):
        ch, cw = u_shaped_model.get_crop_shape(target, reference)
        ref = array(reference)
        tar = array(target)
        _, height_change, width_change = tar - ref
        ch1, ch2 = int(height_change/2), int(height_change/2) + add_factor
        cw1, cw2 = int(width_change/2), int(width_change/2) + add_factor
        assert ch == (ch1, ch2)
        assert cw == (cw1, cw2)

    @pytest.mark.parametrize("use_plus_plus, results", [(True, ([], [], {0: [], 1: []})), (False, ([], [], {}))])
    def test_init_empty_layers_placeholders(self, mocker, use_plus_plus, results):
        model = self.initialize_model(mocker)
        model.plus_plus = use_plus_plus
        model.blocks = 2
        conv_layers, pool_layers, subconv_layers = model.init_empty_layers_placeholders()
        assert conv_layers == results[0]
        assert pool_layers == results[1]
        assert subconv_layers == results[2]

    @pytest.mark.parametrize(
        "channels, result",
        [(1, Input(shape=(2, 2, 1), name='input_img')), (3, Input(shape=(2, 2, 3), name='input_img'))]
        )
    def test_generate_input(self, channels, result, mocker):
        model = self.initialize_model(mocker)
        model.net_h = 2
        model.net_w = 2
        model.channels = channels
        model.generate_input()
        assert model.generate_input().shape[1:] == result.shape[1:]


    @pytest.mark.parametrize(
        "plus_plus, deep_supervision, result_tensor",
        [(False, False, Resizing(200, 200)(Input((300, 300, 3)))), (True, True, [Resizing(200, 200)(Input((300, 300, 3)))])]
        )
    def test_generate_output(self, plus_plus, deep_supervision, result_tensor, mocker):
        model = self.initialize_model(mocker)
        model.net_h = 200
        model.net_w = 200
        model.plus_plus = plus_plus
        model.deep_supervision = deep_supervision
        output_tensor = Input((300, 300, 3))
        mocker.patch(self.u_shaped_path + ".convolutional_layer", return_value=self.mocked_activation)
        mocker.patch("pyplatypus.segmentation.models.u_shaped_models.Average", return_value=self.mocked_activation)
        assert model.generate_output(output_tensor=output_tensor, subconv_layers={0: []})[0].shape[1:] == result_tensor[0].shape[1:]

    @pytest.mark.parametrize("use_linknet, result_connection", [(True, Add), (False, Concatenate)])
    def test_horizontal_connection(self, use_linknet, result_connection, mocker):
        model = self.initialize_model(mocker)
        model.linknet = use_linknet
        assert model.horizontal_connection() == result_connection

    