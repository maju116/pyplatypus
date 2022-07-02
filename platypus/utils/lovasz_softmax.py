from __future__ import print_function, division

from keras import backend as kb

import tensorflow as tf


class LovaszSoftmaxLoss(object):
    @staticmethod
    def lovasz_grad(gt_sorted: tf.Tensor) -> tf.Tensor:
        """
        Computes gradient of the Lovasz extension, based on sorted errors.

        Parameters
        ----------
        gt_sorted: tf.Tensor
            Absolute errors between the layer associated with the certain class and the probabilities produced by the model.
            The errors must be sorted and descending.

        Returns
        -------
        jaccard: tf.Tensor
            Basically the slightly modified Jaccard index based on the sorted errors.
        """
        gts = tf.reduce_sum(gt_sorted)
        intersection = gts - tf.cumsum(gt_sorted)
        union = gts + tf.cumsum(1. - gt_sorted)
        jaccard = 1. - intersection / union
        jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
        return jaccard

    def lovasz_softmax(self, probas: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:  # TODO Per image, Binary
        """
        Calculates the Lovasz-Softmax loss for the Multi-class case.
        It is important note that the inputs are expected to be of the same format as used within the
        Tensorflow data generator which in case of the segmentation task would be:
        (batch_size, image_height, image_width, number_of_classes) hence for instance to access the whole
        batch for the certain class you would use the index: [:, :, :, class_index].

        Parameters
        ----------
        probas: tf.Tensor
            The tensor containing each class' probabilities.
        labels: tf.Tensor
            Ground truth label probabilities (binary variates).

        Returns
        -------
        loss: tf.Tensor
            Lovasz loss, one-element float tensor.
        """
        loss = self.lovasz_softmax_flat(probas, labels)
        return loss

    def lovasz_softmax_flat(self, probas, labels):
        """
        Calculates the Lovasz-Softmax loss for the Multi-class case.

        Parameters
        ----------
        probas: tf.Tensor
            The tensor containing each class' probabilities.
        labels: tf.Tensor
            Ground truth label probabilities (binary variates).

        Returns
        -------
        loss: tf.Tensor
            Lovasz loss, one-element float tensor.
        """
        C = probas.shape[3]
        losses = []
        class_to_sum = list(range(C))
        for c in class_to_sum:
            fg = kb.flatten(tf.cast(labels[:, :, :, c], probas.dtype))  # TODO To do images separately, the additional loop or fn_map is needed.
            class_prob = kb.flatten(probas[:, :, :, c])
            # Calculate the current class prediction errors, probablity based.
            errors = tf.abs(fg - class_prob)
            errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
            fg_sorted = tf.gather(fg, perm)
            grad = self.lovasz_grad(fg_sorted)  # TODO Equals fg sorted, why???
            losses.append(
                tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                        )
        losses_tensor = tf.stack(losses)
        loss = tf.reduce_mean(losses_tensor)
        return loss
