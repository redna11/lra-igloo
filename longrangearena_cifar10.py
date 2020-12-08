# Copyright (c) 2020 ReDNA Labs Co., Ltd.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Lambda, LSTM, Dropout, BatchNormalization
from tensorflow.keras import Model

from mandarin_common_tf2 import *

AUTOTUNE = tf.data.experimental.AUTOTUNE


##############################################################
# Location
##############################################################

print("---------------------------------------------------------")
print("USE WITH TF 2.0")
print("---------------------------------------------------------")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'], with_info=True,
)

print("ds_info", ds_info)


def decode(x):
    decoded = {
        'inputs':
            tf.cast(tf.image.rgb_to_grayscale(x['image']), dtype=tf.int32),
        'targets':
            x['label']
    }

    normalize = True
    if normalize:
        decoded['inputs'] = decoded['inputs'] / 255
    return (decoded['inputs'], decoded['targets'])


batch_size = 128

ds_train = ds_train.map(decode, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.map(decode, num_parallel_calls=AUTOTUNE)

ds_train = ds_train.repeat()
ds_train = ds_train.batch(batch_size, drop_remainder=True)
ds_test = ds_test.batch(batch_size, drop_remainder=True)

ds_train = ds_train.shuffle(
    buffer_size=256, reshuffle_each_iteration=True)


nb_patches = 4000
nb_filters_conv1d = 5
l2_reg = 0.000001
dr = 0.25
nb_stacks = 1
transformer_style = False


inputs = tf.keras.layers.Input(shape=(32, 32, 1,), name="inputs")

x = Lambda(lambda dd: tf.reshape(dd, [-1, 32*32, 1]))(inputs)
x = IGLOO1D_BLOCK(x, nb_patches, nb_filters_conv1d, conv1d_kernel=1, patch_size=4,
                  nb_stacks=nb_stacks, DR=dr, l2_reg=l2_reg, transformer_style=transformer_style)

x = BatchNormalization()(x)
x = Dropout(dr)(x)

targets = Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=targets, name="cifar10_model")


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

print(model.summary())

model.fit(
    ds_train,
    epochs=300,
    validation_data=ds_test,
    steps_per_epoch=50000/batch_size
)
