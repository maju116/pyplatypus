from pyplatypus.utils.toolbox import (
    convert_to_snake_case, split_masks_into_binary, concatenate_binary_masks,
    sum_multiclass_masks, transform_probabilities_into_binaries, binary_based_on_arg_max
    )
import tensorflow as tf
import numpy as np
import pytest


test_cases = ["Dice loss", "DiCe LOSS", "dice loss"]
@pytest.mark.parametrize("any_case", test_cases)
def test_convert_to_snake_case(any_case):
    assert convert_to_snake_case(any_case) == "dice_loss"


colormaps = [[(255, 255, 255), (0, 0, 0)], [(255, 255, 255), (111, 111, 111)], [(255, 255, 255), (111, 111, 111), (0, 0, 0)]]
multiclass_masks = [np.array([255, 255, 255, 111, 111, 111, 0, 0, 0, 0, 0, 0]).reshape(2, 2, 3)]*3
results = [
    np.array([1, 0, 0, 0, 0, 1, 0, 1]).reshape(2, 2, 2),
    np.array([1, 0, 0, 1, 0, 0, 0, 0]).reshape(2, 2, 2),
    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]).reshape(2, 2, 3)
    ]

test_data_and_result = [
    (colormaps[0], multiclass_masks[0], results[0]),
    (colormaps[1], multiclass_masks[1], results[1]),
    (colormaps[2], multiclass_masks[2], results[2])
]

@pytest.mark.parametrize("colormap, mask, result", test_data_and_result)
def test_split_masks_into_binary(colormap, mask, result):
    assert (split_masks_into_binary(mask, colormap) == result).all()


@pytest.mark.parametrize("colormap, result, binary_mask", test_data_and_result)
def test_concatenate_binary_masks(colormap, result, binary_mask):
    classes = [colour[0] for colour in colormap]
    binary_masks_list = concatenate_binary_masks(binary_mask, colormap)
    for i in range(len(binary_masks_list)):
        trimmed_result = np.where(result[:, :, i] == classes[i], result[:, :, i], 0)
        assert (binary_masks_list[i].reshape(2, 2) == trimmed_result).all()

def test_sum_multiclass_masks():
    multiclass_masks_list = [
        np.array([255, 0, 0, 0]).reshape(2, 2, 1), np.array([0, 0, 0, 0]).reshape(2, 2, 1)
    ]
    colormap = [(255, 255, 255), (0, 0, 0)]
    expected = np.array([255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(2, 2, 3)
    assert (sum_multiclass_masks(multiclass_masks_list, colormap) == expected).all()

binaries_data = [
    (np.array([.3, .4]), np.array([0, 1])),
    (np.array([.4, .3]), np.array([1, 0])),
    (np.array([.4, .4]), np.array([1, 1]))
]
@pytest.mark.parametrize("raw, binary", binaries_data)
def test_binary_based_on_arg_max(raw, binary):
    assert (binary_based_on_arg_max(raw) == binary).all() 

def test_transform_probabilities_into_binaries(mocker):
    prediction = np.array([.4, .5, .4, .2, .4, .5, .4, .2]).reshape(2, 2, 2)
    mocker.patch("pyplatypus.utils.toolbox.binary_based_on_arg_max", return_value=1)
    assert (transform_probabilities_into_binaries(prediction) == np.array([1, 1, 1, 1]).reshape(2, 2, 1)).all()
