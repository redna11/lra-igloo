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

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Embedding, LSTM, BatchNormalization
from tensorflow.keras import Model

from mandarin_common_tf2 import *
from data import input_pipeline


import sys
sys.path.append('/home/sieva/dev-python/Tensor-Flow-code/igloo-tf2/')


##############################################################
# Location
##############################################################

print("---------------------------------------------------------")
print("USE WITH TF 2.0")
print("---------------------------------------------------------")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


ds_size = 147086  # CHANGE HERE CHANGE HERE CHANGE HERE CHANGE HERE CHANGE HERE


bs = 40
max_length = 4000
cut_or_pad = max_length


train_ds, eval_ds, test_ds, encoder = input_pipeline.get_matching_datasets(
    n_devices=1,
    task_name="matching",
    data_dir="./data",
    batch_size=bs,
    fixed_vocab=None,
    max_length=max_length,
    tokenizer="char",
    vocab_file_path="")


vocab_size = encoder.vocab_size
train_ds = train_ds.repeat()


nb_patches = 970

embedding_dim = 128
nb_filters_conv1d = 128

l2_reg = 0.01
dr = 0.20  # 0.15
nb_stacks = 1
pooling = 8
patch_size = 4


##################################################
# Define model
##################################################

inputs1 = tf.keras.layers.Input(shape=(2*max_length), name="inputs1")


inputs_a = Lambda(lambda dd: dd[:, :max_length])(inputs1)
inputs_b = Lambda(lambda dd: dd[:, max_length:])(inputs1)

myembedding = Embedding( input_dim=257, output_dim=embedding_dim, trainable=True)

x_a = myembedding(inputs_a)
x_b = myembedding(inputs_b)

my_encoder_a = IGLOO1D_BLOCK(x_a, nb_patches, nb_filters_conv1d, patch_size=patch_size,
                                             nb_stacks=nb_stacks, DR=dr, l2_reg=l2_reg, transformer_style=True, pooling_size=pooling)
my_encoder_b = IGLOO1D_BLOCK(x_b, nb_patches, nb_filters_conv1d, patch_size=patch_size,
                                             nb_stacks=nb_stacks, DR=dr, l2_reg=l2_reg, transformer_style=True, pooling_size=pooling)





x = Lambda(lambda dd: tf.concat(
    [dd[0], dd[1], dd[0]-dd[1], dd[0]*dd[1]], axis=-1))([my_encoder_a, my_encoder_b])


x = Dense(128)(x)
x = Activation("relu")(x)
x = Dropout(dr)(x)

x = Dense(2, activation="softmax")(x)

model = Model(inputs=[inputs1], outputs=x, name="matching_model")


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.0002, decay=0.0005),
    metrics=['accuracy'],
)


model.fit(
    train_ds,
    steps_per_epoch=int(ds_size / bs),
    epochs=20,
    validation_data=test_ds,
)
