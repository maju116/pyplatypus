import pytest
import numpy as np
import pandas as pd
from platypus.detection.utils import yolo3_predict
from platypus.detection.models.yolo3 import coco_anchors, coco_labels
import pickle

with open('testdata/detection/yolo3_test_predictions.pickle', 'rb') as handle:
    test_img_preds = pickle.load(handle)
test_y3p = yolo3_predict(test_img_preds, coco_anchors, coco_labels, image_h=416, image_w=416)


@pytest.mark.parametrize("x, y",
                         [(0, 0.5), (-0.3, 0.425557483188341),
                          (0.2, 0.549833997312478), (11, 0.999983298578152)])
def test_sigmoid(x, y):
    assert test_y3p.sigmoid() == y


@pytest.mark.parametrize("x, y",
                         [(0.1, -2.197224577336219), (0.5, 0), (0.77, 1.2083112059245342)])
def test_logit(x, y):
    assert test_y3p.logit() == y
