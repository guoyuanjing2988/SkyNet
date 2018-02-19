from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

from os import listdir
from os.path import join

import random
import numpy as np
import threading

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from skynet.util.image_processing import get_images
from skynet.util.image_processing import plot_image
from skynet.util.autoencoder import autoencoder

dataset_path = "../data/"
HEIGHT = 300
WIDTH = 300
CHANNEL = 3
DIMENSIONS = HEIGHT * WIDTH * CHANNEL

all_filepaths = [dataset_path + fp for fp in listdir(dataset_path)]

all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)

test_set_size = int(0.2 * len(all_filepaths))
paritions = [0] * len(all_filepaths)
paritions[: test_set_size] = [1] * test_set_size
random.shuffle(paritions)


train_images, test_images = tf.dynamic_partition(all_images, paritions, 2)

train_input_queue = tf.train.slice_input_producer([train_images], shuffle=False)
test_input_queue = tf.train.slice_input_producer([test_images], shuffle=False)

file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_png(file_content, channels=CHANNEL)

file_content = tf.read_file(train_input_queue[0])
test_image = tf.image.decode_png(file_content, channels=CHANNEL)


train_image = tf.reshape(train_image, [DIMENSIONS, 1])
test_image = tf.reshape(test_image, [DIMENSIONS, 1])

BATCH_SIZE = 100
train_image_batch = tf.train.batch([train_image], batch_size=BATCH_SIZE)
test_image_batch = tf.train.batch([test_image], batch_size=BATCH_SIZE)

# Autoencoder
learning_rate = 0.01
l2_reg = 0.0001
num_steps = 500

display_step = 100

num_inputs = DIMENSIONS
num_hidden_1 = 1024
num_hidden_2 = 512
num_hidden_3 = 256
num_hidden_4 = 128
num_hidden_5 = num_hidden_3
num_hidden_6 = num_hidden_2
num_hidden_7 = num_hidden_1
num_hidden_8 = num_inputs


activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder("float", [BATCH_SIZE, num_inputs])

weights1_init = initializer([num_inputs, num_hidden_1])
weights2_init = initializer([num_hidden_1, num_hidden_2])
weights3_init = initializer([num_hidden_2, num_hidden_3])
weights4_init = initializer([num_hidden_3, num_hidden_4])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
weights4 = tf.Variable(weights4_init, dtype=tf.float32, name="weights4")

weights5 = tf.transpose(weights4, name="weights5")
weights6 = tf.transpose(weights3, name="weights6")
weights7 = tf.transpose(weights2, name="weights7")
weights8 = tf.transpose(weights1, name="weights8")

biases1 = tf.Variable(tf.zeros(num_hidden_1), name="biases1")
biases2 = tf.Variable(tf.zeros(num_hidden_2), name="biases2")
biases3 = tf.Variable(tf.zeros(num_hidden_3), name="biases3")
biases4 = tf.Variable(tf.zeros(num_hidden_4), name="biases4")
biases5 = tf.Variable(tf.zeros(num_hidden_5), name="biases5")
biases6 = tf.Variable(tf.zeros(num_hidden_6), name="biases6")
biases7 = tf.Variable(tf.zeros(num_hidden_7), name="biases7")
biases8 = tf.Variable(tf.zeros(num_hidden_8), name="biases8")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
hidden4 = activation(tf.matmul(hidden3, weights4) + biases4)
hidden5 = activation(tf.matmul(hidden4, weights5) + biases5)
hidden6 = activation(tf.matmul(hidden5, weights6) + biases6)
hidden7 = activation(tf.matmul(hidden6, weights7) + biases7)
outputs = tf.matmul(hidden7, weights8) + biases8

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_loss = regularizer(weights1) + regularizer(weights2) + regularizer(weights3) + regularizer(weights4)
loss = reconstruction_loss + reg_loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1, num_steps + 1):

        batch_x = np.reshape(sess.run(train_image_batch), (BATCH_SIZE, DIMENSIONS))
        _, l = sess.run([optimizer, loss], feed_dict={X : batch_x})

        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' %(i, l))
            save_path = saver.save(sess, "/tmp/model" + str(i) +  ".ckpt")
            print("Model saved in path: %s" % save_path)

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
    sess.close()
