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



import functools
import itertools
import os
import time
from absl import app
from absl import flags
import input_pipeline
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

from tensorflow.keras.layers import Dense, Flatten, Conv2D,Embedding,LSTM
from tensorflow.keras import Model


from mandarin_common_tf2 import *


print("---------------------------------------------------------")
print("USE WITH TF 2.0")
print("---------------------------------------------------------")


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"



output_dir_train = 'lra_listops/listops/'


bs=36
cutoff = 2000
embedding_dim = 512



#  tf.enable_v2_behavior()
parent_dir = os.getcwd()
listops_dir = os.path.join(parent_dir, output_dir_train)
os.makedirs(listops_dir, exist_ok=True)
print("-----------dataset-setting--------------")

train_ds, eval_ds, test_ds, encoder = input_pipeline.get_datasets(
  n_devices=1,
  task_name="basic",
  data_dir=listops_dir,
  batch_size=bs,
  max_length=cutoff)


vocab_size = encoder.vocab_size
train_ds = train_ds.repeat()


#############################################################################
####
#### DEFINING the network
####
#############################################################################

nb_patches=2000
nb_filters_conv1d = 512
l2_reg = 0.02
dr=0.5
nb_stacks = 2


inputs = tf.keras.layers.Input(shape=(cutoff,))

x = Embedding(input_dim=16,output_dim=embedding_dim,trainable=True)(inputs)

x = IGLOO1D_BLOCK(x,nb_patches,nb_filters_conv1d,nb_stacks=nb_stacks,DR=dr,l2_reg=l2_reg,transformer_style=True)

targets = Dense(10, activation="softmax")(x)


model = Model(inputs=inputs, outputs=targets, name="blae")


model.compile(
    loss=['sparse_categorical_crossentropy'],
    optimizer=tf.keras.optimizers.Adam(0.0005),
    metrics=['accuracy'],
)

print(model.summary())

model.fit(
    train_ds,
    epochs=60,
    validation_data=test_ds,
    steps_per_epoch=97500/bs


)
