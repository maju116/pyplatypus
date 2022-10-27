import pytest
import numpy as np
from pyplatypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationData, SemanticSegmentationModelSpec
from pyplatypus.segmentation.generator import SegmentationGenerator, split_masks_into_binary, prepare_data_generator, predict_from_generator


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
    assert SegmentationGenerator.create_images_masks_paths(path, mode, only_images, ("images", "masks"), ";") == result


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
    "path, colormap, mode, net_h, net_w, h_splits, w_splits, batch_size, result",
    [
        ("tests/testdata/nested_dirs",
         [(0, 0, 0), (111, 111, 111), (222, 222, 222), (255, 255, 255)],
         "nested_dirs", 2, 2, 1, 1, 3,
         (
                 np.stack([
                     np.array([255, 255, 255,
                               0, 0, 0,
                               111, 111, 111,
                               222, 222, 222]).reshape((2, 2, 3)),
                     np.array([255, 255, 255,
                               222, 222, 222,
                               111, 111, 111,
                               0, 0, 0]).reshape((2, 2, 3)),
                     np.array([222, 222, 222,
                               0, 0, 0,
                               111, 111, 111,
                               255, 255, 255]).reshape((2, 2, 3))
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
         )),
        ("tests/testdata/nested_dirs",
         [(0, 0, 0), (111, 111, 111), (222, 222, 222), (255, 255, 255)],
         "nested_dirs", 2, 2, 2, 2, 3,
         (
                 np.stack([
                     np.array([255, 255, 255,
                               191, 191, 191,
                               219, 219, 219,
                               178, 178, 178]).reshape((2, 2, 3)),
                     np.array([63, 63, 63,
                               0, 0, 0,
                               96, 96, 96,
                               55, 55, 55]).reshape((2, 2, 3)),
                     np.array([147, 147, 147,
                               151, 151, 151,
                               111, 111, 111,
                               138, 138, 138]).reshape((2, 2, 3)),
                     np.array([161, 161, 161,
                               166, 166, 166,
                               194, 194, 194,
                               222, 222, 222]).reshape((2, 2, 3)),
                     np.array([255, 255, 255,
                               246, 246, 246,
                               219, 219, 219,
                               205, 205, 205]).reshape((2, 2, 3)),
                     np.array([230, 230, 230,
                               222, 222, 222,
                               179, 179, 179,
                               166, 166, 166]).reshape((2, 2, 3)),
                     np.array([147, 147, 147,
                               124, 124, 124,
                               111, 111, 111,
                               83, 83, 83]).reshape((2, 2, 3)),
                     np.array([78, 78, 78,
                               55, 55, 55,
                               27, 27, 27,
                               0, 0, 0]).reshape((2, 2, 3)),
                     np.array([222, 222, 222,
                               166, 166, 166,
                               194, 194, 194,
                               161, 161, 161]).reshape((2, 2, 3)),
                     np.array([55, 55, 55,
                               0, 0, 0,
                               96, 96, 96,
                               63, 63, 63]).reshape((2, 2, 3)),
                     np.array([138, 138, 138,
                               151, 151, 151,
                               111, 111, 111,
                               147, 147, 147]).reshape((2, 2, 3)),
                     np.array([178, 178, 178,
                               191, 191, 191,
                               219, 219, 219,
                               255, 255, 255]).reshape((2, 2, 3))
                 ], axis=0),
                 np.stack([
                     np.array([0, 0, 0, 1,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 0, 0,
                               1, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 0, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 1, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 0, 1,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 0, 0,
                               0, 0, 1, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 0, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               1, 0, 0, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 1, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 0, 0,
                               1, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 0, 0]).reshape((2, 2, 4)),
                     np.array([0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 1]).reshape((2, 2, 4))
                 ], axis=0)
         ))
    ]
)
def test_segmentation_generator(path, colormap, mode, net_h, net_w, h_splits, w_splits, batch_size, result):
    test_sg = SegmentationGenerator(path, colormap, mode, False, net_h, net_w, None, None, h_splits, w_splits, 3, None,
                                     batch_size, False, ("images", "masks"), ";")
    output = test_sg.__getitem__(0)
    assert np.allclose(output[0], result[0])
    assert (output[1] == result[1]).all()

def test_calculate_steps_per_epoch(mocker):
    mocker.patch("pyplatypus.segmentation.generator.SegmentationGenerator.create_images_masks_paths", return_value={"images_paths": ["path1", "path2", "path3", "path4"]})
    mocker.patch("pyplatypus.segmentation.generator.SegmentationGenerator.on_epoch_end", return_value=None)
    test_sg = SegmentationGenerator(path="", colormap=[])
    test_sg.batch_size = 2
    assert test_sg.calculate_steps_per_epoch() == 2

def test_prepare_data_generator(mocker):
    data = SemanticSegmentationData(**{
        "train_path": "tests/",
        "validation_path": "tests/",
        "colormap": [[0,0,0]],
        "mode": "nested_dirs",
        "shuffle": True,
        "subdirs": ["images", "masks"],
        "column_sep": ","
        })
    model_cfg = SemanticSegmentationModelSpec(**{
        "name": "model_name",
        "net_h": 8,
        "net_w": 8,
        "blocks": 4,
        "n_class": 2,
        "filters": 2,
        "dropout": .1
    })
    mocker.patch("pyplatypus.segmentation.generator.SegmentationGenerator", return_value="semantic_generator")
    assert prepare_data_generator(data, model_cfg) == "semantic_generator"


class mocked_model:
    def predict(images_batch: list):
        return [images_batch + "_predicted_mask"]

class mocked_generator:
    def __init__(self, max=3):
        self.n = 1
        self.max = max
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.n > self.max:
            raise StopIteration

        result_img = f"img_{self.n}"
        result_path = [[f"path_{self.n}"]]
        self.n += 1
        return result_img, result_path


def test_predict_from_generator():
    predictions, paths = predict_from_generator(model=mocked_model, generator=mocked_generator())
    assert set(predictions) == set(["img_1_predicted_mask", "img_2_predicted_mask", "img_3_predicted_mask"])
    assert paths == ["path_1", "path_2", "path_3"]
