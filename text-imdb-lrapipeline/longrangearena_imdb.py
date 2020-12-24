import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import sys

import input_pipeline

from tensorflow.keras.layers import Dense, Flatten, Conv2D,Embedding,LSTM
from tensorflow.keras import Model



sys.path.append(os.path.normpath(os.path.dirname(os.getcwd())))
from mandarin_common_tf2 import *


##############################################################
###Location
##############################################################

print("---------------------------------------------------------")
print("USE WITH TF 2.0")
print("---------------------------------------------------------")


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



bs = 50
cutoff=4000
cut_or_pad=cutoff



ds_train, _, ds_test, mencoder = input_pipeline.get_tc_datasets(
  n_devices=1,
  task_name='imdb_reviews',
  data_dir=None,
  batch_size=bs,
  fixed_vocab=None,
  max_length=cutoff)



inputs = tf.keras.layers.Input(shape=(cutoff))

nb_patches=1300             
nb_filters_conv1d = 256
l2_reg = 0.005               
dr=0.25                     
nb_stacks = 2
incoming_proj = 24
pooling_size = 4

x = inputs
x = Embedding(input_dim=257,output_dim=256,trainable=True)(x)

x = IGLOO1D_BLOCK(x,nb_patches,nb_filters_conv1d,patch_size=4,nb_stacks=nb_stacks,DR=dr,l2_reg=l2_reg,transformer_style=True,pooling_size=pooling_size,blockstyle="v1",incoming_proj=incoming_proj)

x = Dense(80)(x)
x = Activation("relu")(x)
x = Dropout(dr)(x)


x = Dense(2, activation="softmax")(x)

model = Model(inputs=inputs, outputs=x, name="igloo_model")


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.00015),
    metrics=['accuracy'],
)

print(model.summary())

model.fit(
    ds_train,
    epochs=120,
    validation_data=ds_test,
)
