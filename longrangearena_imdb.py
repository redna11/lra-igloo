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
import sys

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Embedding, LSTM
from tensorflow.keras import Model

from mandarin_common_tf2 import *


##############################################################
# Location
##############################################################

print("---------------------------------------------------------")
print("USE WITH TF 2.0")
print("---------------------------------------------------------")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


(ds_train, ds_test), ds_info = tfds.load(
    'imdb_reviews/bytes',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

print(ds_info)

bs = 128
cutoff = 4000
cut_or_pad = cutoff


ds_train = ds_train.map(lambda x, y: (x[:cutoff], y))
ds_train = ds_train.padded_batch(bs,  padded_shapes=([cutoff], []))
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


ds_test = ds_test.map(lambda x, y: (x[:cutoff], y))
ds_test = ds_test.padded_batch(bs, padded_shapes=([cutoff], []))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


inputs = tf.keras.layers.Input(shape=(cutoff))

nb_patches = 650
nb_filters_conv1d = 256
l2_reg = 0.005
dr = 0.10
nb_stacks = 1

x = inputs
x = Embedding(input_dim=257, output_dim=256, trainable=True)(x)

x = IGLOO1D_BLOCK(x, nb_patches, nb_filters_conv1d, patch_size=4,
                  nb_stacks=nb_stacks, DR=dr, l2_reg=l2_reg, transformer_style=True, v2=False)

x = Dense(80)(x)
x = Activation("relu")(x)
x = Dropout(dr)(x)


x = Dense(2, activation="softmax")(x)

model = Model(inputs=inputs, outputs=x, name="mnist_model")


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.0005),
    metrics=['accuracy'],
)

print(model.summary())

model.fit(
    ds_train,
    epochs=100,
    validation_data=ds_test,
)
