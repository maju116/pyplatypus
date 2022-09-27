from pyplatypus.utils.lovasz_softmax import LovaszSoftmaxLoss as LSL
import tensorflow as tf
import numpy as np

import pytest


class TestLovaszSoftmax:
    y_actual = tf.constant(
        np.array([
            0, 0, 0, 1,
            1, 0, 0, 0,
            ]).reshape((1, 2, 2, 2)))

    y_pred = tf.constant(
        np.array([
            0, 0, 0, 0.5,
            0.5, 0, 0, 0,
            ]).reshape((1, 2, 2, 2)), tf.float32)
    lsl_path = "pyplatypus.utils.lovasz_softmax.LovaszSoftmaxLoss"
    lsl = LSL()

    def mocked_lovasz_softmax_batch(self, probas, labels):
        assert len(probas.shape) == 4
        assert len(labels.shape) == 4
        return 0

    def test_lovasz_grad(self):
        gt_sorted = tf.constant([2, 1], dtype=tf.float32)
        result = self.lsl.lovasz_grad(gt_sorted)
        # Mimicing the expected cacluations inside the function
        intersection = 2 + 1 - np.array((2, 2+1))
        union = 2 + 1 + np.array((1-2, -1))
        jaccard = 1 - intersection / union
        jaccard = np.concatenate((jaccard[0:1], jaccard[:-1]), axis=0)
        assert all(result == jaccard)

    def test_lovasz_softmax_batch(self, mocker):
        mocker.patch(self.lsl_path + ".lovasz_softmax", self.mocked_lovasz_softmax_batch)
        y_actual = tf.reshape(self.y_actual, (2, 2, 2))
        y_pred = tf.reshape(self.y_pred, (2, 2, 2))
        result = self.lsl.lovasz_softmax_batch(probas_labels=(y_actual, y_pred))
        assert result == 0

    def test_lovasz_softmax(self, mocker):
        mocker.patch(self.lsl_path + ".lovasz_softmax_flat", return_value=0)
        result = self.lsl.lovasz_softmax(probas=self.y_pred, labels=self.y_actual)
        assert result == 0

    def test_flatten_and_select(self):
        res1, res2 = self.lsl.flatten_and_select(
            labels=self.y_actual,
            probas=self.y_pred,
            index=1)
        len_ref = self.y_actual.shape[1] + self.y_actual.shape[2]
        assert len(res1) == len_ref
        assert len(res2) == len_ref

    def test_calculate_and_sort_errors(self):
        fg = tf.constant([1, 1], dtype=tf.float64)
        class_prob = tf.constant([.5, 0], dtype=tf.float64)
        fg_sorted, errors, errors_sorted = self.lsl.calculate_and_sort_errors(fg, class_prob)
        assert all(errors == (.5, 1))
        assert all(errors_sorted == (1, .5))
        assert all(fg_sorted == fg)

    def test_lovasz_softmax_flat(self, mocker):
        mocker.patch(self.lsl_path + ".flatten_and_select", return_value=(None, None))
        mocker.patch(
            self.lsl_path + ".calculate_and_sort_errors", return_value=(
                None, None, tf.constant([1, 0], dtype=tf.float64)
                )
            )
        mocker.patch(
            self.lsl_path + ".lovasz_grad", return_value=tf.constant([1, 0], dtype=tf.float64)
            )
        assert self.lsl.lovasz_softmax_flat(probas=self.y_pred, labels=self.y_actual) == 1
