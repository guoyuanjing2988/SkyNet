from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def autoencoder(x, weights, biases):

    if (len(weights) != len(biases)):
        raise AssertionError("Number of Weights must be equal to the number of biases")

    layers = []
    count = 0

    for weight, biase in zip(weights, biases):
        if count == 0:
            layers.append(tf.nn.sigmoid(tf.add(tf.matmul(x, weight), biase)))
        else:
            layers.append(tf.nn.sigmoid(tf.add(tf.matmul(layers[count - 1], weight), biase)))
        count += 1
    return layers[-1]
