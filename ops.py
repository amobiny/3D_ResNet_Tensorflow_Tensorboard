"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     8/1/2017
Comments: Includes functions for defining the 3D-ResNet layers
**********************************************************************************
"""

import tensorflow as tf
from tflearn.layers.normalization import batch_normalization


def weight_variable(name, shape):
    """Create a weight variable with appropriate initialization."""
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def new_conv_layer(inputs,  # The previous layer.
                   layer_name,
                   stride,
                   num_inChannel,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   batch_norm,
                   use_relu):
    """Create a convolution layer."""
    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, filter_size, num_inChannel, num_filters]
    with tf.variable_scope(layer_name):
        weights = weight_variable(layer_name, shape=shape)
        # tf.summary.histogram('histogram', weights)
        biases = bias_variable(layer_name, [num_filters])
        layer = tf.nn.conv3d(input=inputs,
                             filter=weights,
                             strides=[1, stride, stride, stride, 1],
                             padding="SAME")
        if batch_norm:
            layer = batch_normalization(layer)
        # Add the biases to the results of the convolution.
        layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
    return layer


def flatten_layer(layer):
    """Flattens the output of the convolutional layer to prepare it to be fed in to the fully-connected layer"""
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:5].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def fc_layer(bottom,  # The previous layer.
             out_dim,  # Num. outputs.
             name,  # layer name
             batch_norm=False,
             add_reg=True,
             use_relu=True):  # Use Rectified Linear Unit (ReLU)?
    """Create a fully connected layer"""
    in_dim = bottom.get_shape()[1]
    with tf.variable_scope(name):
        weights = weight_variable(name, shape=[in_dim, out_dim])
        # tf.summary.histogram('histogram', weights)
        biases = bias_variable(name, [out_dim])
        if add_reg:
            tf.add_to_collection('reg_weights', weights)
        layer = tf.matmul(bottom, weights)
        if batch_norm:
            layer = batch_normalization(layer)
        layer += biases
        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)
    return layer


def max_pool(x, ksize, stride, name):
    """Create a max pooling layer."""
    return tf.nn.max_pool3d(x,
                            ksize=[1, ksize, ksize, ksize, 1],
                            strides=[1, stride, stride, stride, 1],
                            padding="SAME",
                            name=name)


def avg_pool(x, ksize, stride, name):
    """Create an average pooling layer."""
    return tf.nn.avg_pool3d(x,
                            ksize=[1, ksize, ksize, ksize, 1],
                            strides=[1, stride, stride, stride, 1],
                            padding="VALID",
                            name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)


def bottleneck_block(x, num_ch, block_name,
                     s1, k1, nf1, name1,
                     s2, k2, nf2, name2,
                     s3, k3, nf3, name3,
                     s4, k4, name4, first_block=False):
    # num_ch = x.get_shape()[-1]
    with tf.variable_scope(block_name):
        # Convolutional Layer 1
        layer_conv1 = new_conv_layer(inputs=x,
                                     layer_name=name1,
                                     stride=s1,
                                     num_inChannel=num_ch,
                                     filter_size=k1,
                                     num_filters=nf1,
                                     batch_norm=True,
                                     use_relu=True)

        # Convolutional Layer 2
        layer_conv2 = new_conv_layer(inputs=layer_conv1,
                                     layer_name=name2,
                                     stride=s2,
                                     num_inChannel=nf1,
                                     filter_size=k2,
                                     num_filters=nf2,
                                     batch_norm=True,
                                     use_relu=True)

        # Convolutional Layer 3
        layer_conv3 = new_conv_layer(inputs=layer_conv2,
                                     layer_name=name3,
                                     stride=s3,
                                     num_inChannel=nf2,
                                     filter_size=k3,
                                     num_filters=nf3,
                                     batch_norm=True,
                                     use_relu=False)

        if first_block:
            shortcut = new_conv_layer(inputs=x,
                                      layer_name=name4,
                                      stride=s4,
                                      num_inChannel=num_ch,
                                      filter_size=k4,
                                      num_filters=nf3,
                                      batch_norm=True,
                                      use_relu=False)
            assert (
                shortcut.get_shape().as_list() == layer_conv3.get_shape().as_list()), "Tensor sizes of the two branches are not matched!"
            res = shortcut + layer_conv3
        else:
            res = layer_conv3 + x
            assert (
                x.get_shape().as_list() == layer_conv3.get_shape().as_list()), "Tensor sizes of the two branches are not matched!"
    return tf.nn.relu(res)
