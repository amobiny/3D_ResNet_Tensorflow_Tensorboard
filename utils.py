"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     8/1/2017
Comments: Python utility functions
**********************************************************************************
"""

import numpy as np
import random
import scipy
import tensorflow as tf


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def reformat(x, y, img_size, num_ch, num_class):
    """ Reformats the data to the format acceptable for 3D conv layers"""
    dataset = x.reshape((-1, img_size, img_size, img_size, num_ch)).astype(np.float32)
    labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
    return dataset, labels


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def _random_rotation_3d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).
    Arguments:
    max_angle: `float`. The maximum rotation angle.
    Returns:
    batch of rotated 3D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image1 = np.squeeze(batch[i])
            # rotate along z-axis
            angle = random.uniform(-max_angle, max_angle)
            image2 = scipy.ndimage.interpolation.rotate(image1, angle, mode='nearest', axes=(0, 1), reshape=False)

            # rotate along y-axis
            angle = random.uniform(-max_angle, max_angle)
            image3 = scipy.ndimage.interpolation.rotate(image2, angle, mode='nearest', axes=(0, 2), reshape=False)

            # rotate along x-axis
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image3, angle, mode='nearest', axes=(1, 2), reshape=False)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)


def accuracy_generator(labels_tensor, logits_tensor):
    """
     Calculates the classification accuracy.
     Note that this is for multi-task classification.
    :param labels_tensor: Tensor of correct predictions of size [batch_size, numClasses]
    :param logits_tensor: Predicted scores (logits) by the model.
            It should have the same dimensions as labels_tensor
    :return: accuracy: average accuracy of the batch
    """

    correct_prediction = tf.equal(tf.argmax(logits_tensor, 1), tf.argmax(labels_tensor, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def cross_entropy_loss(labels_tensor, logits_tensor):
    """
     Calculates the cross-entropy loss function for the given parameters.
     Note that this is for multi-task classification problem.
    :param labels_tensor: Tensor of correct predictions of size [batch_size, numClasses]
    :param logits_tensor: Predicted scores (logits) by the model.
            It should have the same dimensions as labels_tensor
    :return: Cross-entropy Loss tensor
    """
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=logits_tensor, labels=labels_tensor)
    loss = tf.reduce_mean(diff)
    return loss



