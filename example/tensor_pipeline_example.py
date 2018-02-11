from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import listdir
from os.path import join

import random

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

sess = tf.InteractiveSession()

dataset_path = "../data/"

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
train_image = tf.image.decode_png(file_content, channels=3)

file_content = tf.read_file(train_input_queue[0])
test_image = tf.image.decode_png(file_content, channels=3)

train_image.set_shape([300, 300, 3])
test_image.set_shape([300, 300, 3])

BATCH_SIZE = 30
train_image_batch = tf.train.batch([train_image], batch_size=BATCH_SIZE)
test_image_batch = tf.train.batch([test_image], batch_size=BATCH_SIZE)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("From the train set:")
    for i in range(20):
        print(sess.run(train_image_batch))

    print("From the test set:")
    for i in range(10):
        print(sess.run(test_image_batch))

    coord.request_stop()
    coord.join(threads)
    sess.close()
