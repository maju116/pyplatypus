from __future__ import print_function, division

from keras import backend as kb

import tensorflow as tf


class LovaszSoftmaxLoss(object):
    """The class makes the computation of Lovasz Softmax Loss possible.
    This implementation follows the logic of [1]_.

    Methods
    -------
    lovasz_grad(gt_sorted: tf.Tensor)
        Computes gradient of the Lovasz extension, based on sorted errors.

    lovasz_softmax_batch(self, probas_labels: tuple)
        Calculates the Lovasz-Softmax loss for the Multi-class case, adapted to accept the tf.map_fn-produced arguments.

    lovasz_softmax(self, probas: tf.Tensor, labels: tf.Tensor)
        Calculates the Lovasz-Softmax loss for the Multi-class case.

    flatten_and_select(self, labels: tf.Tensor, probas: tf.Tensor, index: int)
        Selects the image layer associated with the certain class i.e. index and then flattens the selections.

    calculate_and_sort_errors(self, fg: tf.Tensor, class_prob: tf.Tensor)
        Calculates the absolute errors to then sort them alongside the input flattened tensor.

    lovasz_softmax_flat(self, probas: tf.Tensor, labels: tf.Tensor)
        Calculates the Lovasz-Softmax loss for the Multi-class case.


    References
    ----------
    [1] Maxim Berman, Amal Rannen Triki, Matthew B. Blaschko
    "The Lovasz-Softmax loss: A tractable surrogate for the optimization of the ´
    intersection-over-union measure in neural networks"
    Dept. ESAT, Center for Processing Speech and Images
    KU Leuven, Belgium
    """

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

    def lovasz_softmax_batch(self, probas_labels: tuple) -> tf.Tensor:
        """
        Calculates the Lovasz-Softmax loss for the Multi-class case, adapted to accept the tf.map_fn-produced arguments.

        Parameters
        ----------
        probas_labels: tuple
            Tuple containing the class' probabilities tensor and the one with the truth label probabilities.

        Returns
        -------
        loss: tf.Tensor
            Lovasz loss, one-element float tensor.
        """
        probas, labels = probas_labels
        probas, labels = tf.expand_dims(probas, 0), tf.expand_dims(labels, 0)
        loss = self.lovasz_softmax(probas, labels)
        return loss

    def lovasz_softmax(self, probas: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
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

    def flatten_and_select(self, labels: tf.Tensor, probas: tf.Tensor, index: int) -> tuple:
        """Selects the image layer associated with the certain class i.e. index and then flattens the selections.

        Parameters
        ----------
        probas: tf.Tensor
            The tensor containing each class' probabilities.
        labels: tf.Tensor
            Ground truth label probabilities (binary variates).

        Returns
        -------
        (fg, class_prob): Tuple[tf.Tensor, tf.Tensor]
            Flattened selections.
        """
        fg = kb.flatten(tf.cast(labels[:, :, :, index], probas.dtype))
        class_prob = kb.flatten(probas[:, :, :, index])
        return fg, class_prob

    def calculate_and_sort_errors(self, fg: tf.Tensor, class_prob: tf.Tensor) -> tuple:
        """Calculates the absolute errors to then sort them alongside the input flattened tensor.

        Parameters
        ----------
        fg: tf.Tensor
            Flat tensor, representing the ground truth class.
        class_prob: tf.Tensor
            Flat tensor representing class' probabilities produced by model

        Returns
        -------
        fg_sorted: tf.Tensor
            Sorted input tensor, permutation is the same as the one used for sorting the absolute errors.
        errors: tf.Tensor
            Absolute errors.
        errors_sorted: tf.Tensor
            Sorted absolute errors.
        """
        errors = tf.abs(fg - class_prob)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0])
        fg_sorted = tf.gather(fg, perm)
        return fg_sorted, errors, errors_sorted

    def lovasz_softmax_flat(self, probas: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
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
            fg, class_prob = self.flatten_and_select(labels, probas, index=c)
            # Calculate the current class prediction errors, probablity based.
            fg_sorted, errors, errors_sorted = self.calculate_and_sort_errors(fg, class_prob)
            grad = self.lovasz_grad(fg_sorted)
            losses.append(
                tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1)
                        )
        losses_tensor = tf.stack(losses)
        loss = tf.reduce_mean(losses_tensor)
        return loss
