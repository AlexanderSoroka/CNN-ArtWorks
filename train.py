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
import tensorflow as tf
import time
from tensorflow.python import keras as keras

LOG_DIR = 'logs'
SHUFFLE_BUFFER = 256
BATCH_SIZE = 64
NUM_CLASSES = 50
RESIZE_TO = 128
TRAINSET_SIZE = 7666


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


def create_dataset_iterator(files):
    return tf.data.TFRecordDataset(files)\
        .map(parse_proto_example, num_parallel_calls=2)\
        .shuffle(SHUFFLE_BUFFER)\
        .batch(BATCH_SIZE)\
        .repeat()\
        .make_one_shot_iterator()


def build_model(images):
    input = tf.keras.layers.Input(tensor=images)
    layer = tf.keras.layers.Conv2D(filters=128, kernel_size=3)(input)
    layer = tf.keras.layers.Conv2D(filters=128, kernel_size=3)(layer)
    layer = tf.keras.layers.MaxPool2D()(layer)
    layer = tf.keras.layers.Flatten()(layer)
    output = tf.keras.layers.Dense(50, activation=tf.keras.activations.relu)(layer)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model


def main():
    train_images, train_labels = create_dataset_iterator(glob.glob('data/train-*')).get_next()
    train_labels = tf.one_hot(train_labels, NUM_CLASSES)
    model = build_model(train_images)

    model.compile(
        optimizer=keras.optimizers.RMSprop(lr=0.001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy],
        target_tensors=[train_labels]
    )

    model.fit(
        epochs=30,
        steps_per_epoch=int(TRAINSET_SIZE / BATCH_SIZE),
        verbose=1,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir='{}/{}'.format(LOG_DIR, time.time()),
                write_images=True
            )
        ]
    )


if __name__ == '__main__':
    main()
