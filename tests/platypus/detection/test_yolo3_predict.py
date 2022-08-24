import pytest
import numpy as np
import pandas as pd
from platypus.detection.utils import yolo3_predict
from platypus.detection.models.yolo3 import coco_anchors, coco_labels
import pickle

with open('tests/testdata/detection/yolo3_test_predictions.pickle', 'rb') as handle:
    test_img_preds = pickle.load(handle)
test_y3p = yolo3_predict(test_img_preds, coco_anchors, coco_labels, image_h=416, image_w=416)


@pytest.mark.parametrize("x, y",
                         [(0, 0.5), (-0.3, 0.425557483188341),
                          (0.2, 0.549833997312478), (11, 0.999983298578152)])
def test_sigmoid(x, y):
    assert test_y3p.sigmoid(x) == y


@pytest.mark.parametrize("x, y",
                         [(0.1, -2.197224577336219), (0.5, 0), (0.77, 1.2083112059245342)])
def test_logit(x, y):
    assert test_y3p.logit(x) == y


@pytest.mark.parametrize("predictions, result",
                         [(test_img_preds,
                          [pd.DataFrame({
                              'xmin': (86, 337, 145, 182, 201, 208, 25, 222),
                              'ymin': (298, 315, 292, 251, 225, 225, 230, 237),
                              'xmax': (98, 351, 204, 195, 207, 214, 277, 280),
                              'ymax': (359, 361, 367, 267, 232, 231, 336, 340),
                              'p': (0.950761, 0.958715, 0.999735, 0.841772,
                                    0.837264, 0.820939, 0.999457, 0.954366),
                              'label_id': (0, 0, 2, 2, 2, 2, 5, 6),
                              'label': ('person', 'person', 'car', 'car', 'car', 'car',
                                        'bus', 'train')
                          }),
                              pd.DataFrame({
                                  'xmin': (9, 120, 202),
                                  'ymin': (29, 85, 169),
                                  'xmax': (188, 303, 352),
                                  'ymax': (378, 374, 386),
                                  'p': (0.999947, 0.997230, 0.999533),
                                  'label_id': (22, 22, 22),
                                  'label': ('zebra', 'zebra', 'zebra')
                              })
                          ])])
def test_get_boxes(predictions, result):
    boxes = test_y3p.get_boxes()
    for i in range(len(result)):
        assert boxes[i].loc[:, ["xmin", "ymin", "xmax", "ymax", "label_id", "label"]].equals(
            result[1].loc[:, ["xmin", "ymin", "xmax", "ymax", "label_id", "label"]]
        )
        assert np.allclose(boxes[i].p, result[i].p)
