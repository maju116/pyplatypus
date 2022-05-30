import pytest
import numpy as np
import pandas as pd
from platypus.segmentation.generator import *


@pytest.mark.parametrize(
    "path, mode, only_images, result",
    [
        ("tests/testdata/nested_dirs", "nested_dirs", False,
         {'images_paths': [['tests/testdata/nested_dirs/image_1/images/test_image_1.png'],
                           ['tests/testdata/nested_dirs/image_2/images/test_image_2.png'],
                           ['tests/testdata/nested_dirs/image_3/images/test_image_3.png']],
          'masks_paths': [['tests/testdata/nested_dirs/image_1/masks/test_mask_1_1.png',
                           'tests/testdata/nested_dirs/image_1/masks/test_mask_1_2.png',
                           'tests/testdata/nested_dirs/image_1/masks/test_mask_1_3.png'],
                          ['tests/testdata/nested_dirs/image_2/masks/test_mask_2_1.png',
                           'tests/testdata/nested_dirs/image_2/masks/test_mask_2_2.png',
                           'tests/testdata/nested_dirs/image_2/masks/test_mask_2_3.png'],
                          ['tests/testdata/nested_dirs/image_3/masks/test_mask_3_1.png',
                           'tests/testdata/nested_dirs/image_3/masks/test_mask_3_2.png',
                           'tests/testdata/nested_dirs/image_3/masks/test_mask_3_3.png']]}
         ),
        ("tests/testdata/nested_dirs", "nested_dirs", True,
         {'images_paths': [['tests/testdata/nested_dirs/image_1/images/test_image_1.png'],
                           ['tests/testdata/nested_dirs/image_2/images/test_image_2.png'],
                           ['tests/testdata/nested_dirs/image_3/images/test_image_3.png']]}
         ),
        ("tests/testdata/test_config1.csv", "config_file", False,
         {'images_paths': [['tests/testdata/dir/images/test_image_1.png'],
                           ['tests/testdata/dir/images/test_image_2.png'],
                           ['tests/testdata/dir/images/test_image_3.png']],
          'masks_paths': [['tests/testdata/dir/masks/test_mask_1.png'],
                          ['tests/testdata/dir/masks/test_mask_2.png'],
                          ['tests/testdata/dir/masks/test_mask_3.png']]}
         ),
        ("tests/testdata/test_config1.csv", "config_file", True,
         {'images_paths': [['tests/testdata/dir/images/test_image_1.png'],
                           ['tests/testdata/dir/images/test_image_2.png'],
                           ['tests/testdata/dir/images/test_image_3.png']]}
         ),
        ("tests/testdata/test_config2.csv", "config_file", False,
         {'images_paths': [['tests/testdata/nested_dirs/image_1/images/test_image_1.png'],
                           ['tests/testdata/nested_dirs/image_2/images/test_image_2.png'],
                           ['tests/testdata/nested_dirs/image_3/images/test_image_3.png']],
          'masks_paths': [['tests/testdata/nested_dirs/image_1/masks/test_mask_1_1.png',
                           'tests/testdata/nested_dirs/image_1/masks/test_mask_1_2.png',
                           'tests/testdata/nested_dirs/image_1/masks/test_mask_1_3.png'],
                          ['tests/testdata/nested_dirs/image_2/masks/test_mask_2_1.png',
                           'tests/testdata/nested_dirs/image_2/masks/test_mask_2_2.png',
                           'tests/testdata/nested_dirs/image_2/masks/test_mask_2_3.png'],
                          ['tests/testdata/nested_dirs/image_3/masks/test_mask_3_1.png',
                           'tests/testdata/nested_dirs/image_3/masks/test_mask_3_2.png',
                           'tests/testdata/nested_dirs/image_3/masks/test_mask_3_3.png']]}
         )
    ]
)
def test_create_images_masks_paths(path, mode, only_images, result):
    assert segmentation_generator.create_images_masks_paths(path, mode, only_images, ("images", "masks"), ";") == result


@pytest.mark.parametrize(
    "mask, colormap, result",
    [
        (np.array([255, 255, 255,
                   0, 0, 0,
                   111, 111, 111,
                   222, 222, 222]).reshape((2, 2, 3)),
         [(0, 0, 0), (111, 111, 111), (222, 222, 222), (255, 255, 255)],
         np.array([0, 0, 0, 1,
                   1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0]).reshape((2, 2, 4))
         )
    ]
)
def test_split_masks_into_binary(mask, colormap, result):
    assert (split_masks_into_binary(mask, colormap) == result).all()


@pytest.mark.parametrize(
    "path, colormap, mode, batch_size, result",
    [
        ("tests/testdata/nested_dirs", [(0, 0, 0), (111, 111, 111), (222, 222, 222), (255, 255, 255)], "nested_dirs", 3,
         (
                 np.stack([
                     np.array([255, 255, 255,
                               0, 0, 0,
                               111, 111, 111,
                               222, 222, 222]).reshape((2, 2, 3)) * (1 / 255),
                     np.array([255, 255, 255,
                               222, 222, 222,
                               111, 111, 111,
                               0, 0, 0]).reshape((2, 2, 3)) * (1 / 255),
                     np.array([222, 222, 222,
                               0, 0, 0,
                               111, 111, 111,
                               255, 255, 255]).reshape((2, 2, 3)) * (1 / 255)
                 ], axis=0),
                 np.stack([
                     np.array([0, 0, 0, 1,
                               1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 1, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 0, 1,
                               0, 0, 1, 0,
                               0, 1, 0, 0,
                               1, 0, 0, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 1, 0,
                               1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 0, 1]).reshape((2, 2, 4))
                 ], axis=0)
         ))
    ]
)
def test_segmentation_generator(path, colormap, mode, batch_size, result):
    test_sg = segmentation_generator(path, colormap, mode, False, 2, 2, False, None, batch_size, False,
                                     ("images", "masks"), ";")
    output = test_sg.__getitem__(0)
    assert np.allclose(output[0], output[0])
    assert (output[1] == output[1]).all()
