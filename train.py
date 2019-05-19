"""This module implements data feeding and training loop to create model for classification task over
best-art-works dataset as a lab example for BSU students.
"""

__author__ = 'Alexander Soroka, soroka.a.m@gmail.com'
__copyright__ = """Copyright 2019 Alexander Soroka
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the "Software"), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and 
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions 
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED 
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import glob
import numpy as np
import tensorflow as tf
import time
from tensorflow.python import keras as keras
from tensorflow.python.keras.callbacks import LearningRateScheduler

LOG_DIR = 'logs'
SHUFFLE_BUFFER = 10
BATCH_SIZE = 64
NUM_CLASSES = 50
PARALLEL_CALLS=4
RESIZE_TO = 224
TRAINSET_SIZE = 7666
VALSET_SIZE=851


def parse_proto_example(proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }
    example = tf.parse_single_example(proto, keys_to_features)
    example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)
    example['image'] = tf.image.resize_images(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
    return example['image'], example['image/class/label']


def flip(image, label):
    return tf.image.random_flip_up_down(tf.image.random_flip_left_right(image)), label


def resize(image, label):
    return tf.image.resize_images(image, tf.constant([RESIZE_TO, RESIZE_TO])), label


def color(image, label):
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    return tf.clip_by_value(image, 0, 1), label


def create_dataset_iterator(filenames):
    files = tf.data.Dataset.list_files(filenames)
    return files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=PARALLEL_CALLS))\
        .map(parse_proto_example, num_parallel_calls=PARALLEL_CALLS)\
        .map(resize, num_parallel_calls=PARALLEL_CALLS)\
        .cache()\
        .map(flip, num_parallel_calls=PARALLEL_CALLS)\
        .map(color, num_parallel_calls=PARALLEL_CALLS)\
        .batch(BATCH_SIZE)\
        .apply(tf.data.experimental.shuffle_and_repeat(SHUFFLE_BUFFER))\
        .prefetch(2 * BATCH_SIZE)\
        .make_one_shot_iterator()


def build_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=3),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)
    ])


def learning_rate(epoch):
    if epoch < 150:
        return 0.005
    return 0.0001


def main():
    train_images, train_labels = create_dataset_iterator(glob.glob('data/train-*')).get_next()
    train_labels = tf.one_hot(train_labels, NUM_CLASSES)
    model = build_model()

    model.compile(
        optimizer=keras.optimizers.sgd(lr=0, momentum=0.5),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy],
        target_tensors=[train_labels]
    )

    lr = LearningRateScheduler(learning_rate)

    model.fit(
        train_images, train_labels,
        epochs=200, steps_per_epoch=int(np.ceil(TRAINSET_SIZE / float(BATCH_SIZE))),
        validation_data=create_dataset_iterator(glob.glob('data/validation-*')),
        validation_steps=int(np.ceil(VALSET_SIZE / float(BATCH_SIZE))),
        callbacks=[
            lr,
            tf.keras.callbacks.TensorBoard(
                log_dir='{}/{}'.format(LOG_DIR, time.time()),
                write_images=True
            )
        ]
    )


if __name__ == '__main__':
    main()
