import pytest
from pyplatypus.utils.path import create_images_masks_paths


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
    assert create_images_masks_paths(path, mode, only_images, ("images", "masks"), ";") == result
