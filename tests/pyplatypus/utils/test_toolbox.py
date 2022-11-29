from pyplatypus.utils.toolbox import convert_to_snake_case, convert_to_camel_case
import numpy as np
import pytest


test_cases = ["Dice loss", "DiCe LOSS", "dice loss"]
@pytest.mark.parametrize("any_case", test_cases)
def test_convert_to_snake_case(any_case):
    assert convert_to_snake_case(any_case) == "dice_loss"

test_cases = ["Dice loss", "DiCe LOSS", "dice loss"]
@pytest.mark.parametrize("any_case", test_cases)
def convert_to_camel_case(any_case):
    assert convert_to_snake_case(any_case) == "DiceLoss"
