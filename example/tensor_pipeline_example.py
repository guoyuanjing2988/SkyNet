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

from skynet.util.autoencoder import autoencoder

sess = tf.InteractiveSession()

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

BATCH_SIZE = 30
train_image_batch = tf.train.batch([train_image], batch_size=BATCH_SIZE)
test_image_batch = tf.train.batch([test_image], batch_size=BATCH_SIZE)

# Autoencoder
learning_rate = 0.01
num_steps = 500

display_step = 100
examples_to_show = 10

num_input = DIMENSIONS
num_hidden_1 = 1024
num_hidden_2 = 512
num_hidden_3 = 256
num_hidden_4 = 128


encoder_weights = [
    tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3])),
    tf.Variable(tf.random_normal([num_hidden_3, num_hidden_4]))
]

encoder_biases = [
    tf.Variable(tf.random_normal([BATCH_SIZE, num_hidden_1])),
    tf.Variable(tf.random_normal([BATCH_SIZE, num_hidden_2])),
    tf.Variable(tf.random_normal([BATCH_SIZE, num_hidden_3])),
    tf.Variable(tf.random_normal([BATCH_SIZE, num_hidden_4]))
]

decoder_weights = [
    tf.Variable(tf.random_normal([num_hidden_4, num_hidden_3])),
    tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2])),
    tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    tf.Variable(tf.random_normal([num_hidden_1, num_input]))
]

decoder_biases = [
    tf.Variable(tf.random_normal([BATCH_SIZE, num_hidden_3])),
    tf.Variable(tf.random_normal([BATCH_SIZE, num_hidden_2])),
    tf.Variable(tf.random_normal([BATCH_SIZE, num_hidden_1])),
    tf.Variable(tf.random_normal([BATCH_SIZE, num_input]))
]

X = tf.placeholder("float", [BATCH_SIZE, num_input])
encoder_op = autoencoder(X, encoder_weights, encoder_biases)
decoder_op = autoencoder(encoder_op, decoder_weights, decoder_biases)
y_pred = decoder_op
y_true = X
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

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

    # test_data = img_df.loc[399:,'Vector'].values
    # g = sess.run(decoder_op, feed_dict={X: test_data})
    #
    # canvas_orig = np.empty((640, 400, 3))
    # canvas_recon = np.empty((640, 400, 3))
    #
    # canvas_orig = test_data[0].reshape([640, 400, 3])
    # canvas_recon = g[0].reshape([640, 400, 3])
    #
    # print("Original Images")
    # plt.figure(figsize=(640, 400))
    # plt.imshow(canvas_orig)
    # plt.show()
    #
    # print("Reconstructed Images")
    # plt.figure(figsize=(640, 400))
    # plt.imshow(canvas_recon)
    # plt.show()

    coord.request_stop()
    coord.join(threads)
    sess.close()
