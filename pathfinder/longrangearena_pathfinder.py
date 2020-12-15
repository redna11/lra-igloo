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
import  os
import sys


from tensorflow.keras.layers import Dense, Flatten, Conv2D,Lambda,LSTM,Dropout,BatchNormalization
from tensorflow.keras import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

AUTOTUNE = tf.data.experimental.AUTOTUNE

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

batch_size = 40
nb_classes = 10
nb_epoch = 600
data_augmentation = True


################################
#### IGLO params
################################

nb_patches=4000
nb_filters_conv1d = 5
l2_reg = 0.000001
dr=0.25
nb_stacks = 1
transformer_style=False

## setting CNN kernel to 1

inputs = tf.keras.layers.Input(shape=(32,32,1,),name="inputs")

x = Lambda(lambda dd: tf.reshape(dd, [-1,32*32,1]) )(inputs)
x = IGLOO1D_BLOCK(x,nb_patches,nb_filters_conv1d,conv1d_kernel=1,patch_size=4,nb_stacks=nb_stacks,DR=dr,l2_reg=l2_reg,transformer_style=transformer_style)

x=BatchNormalization() (x)
x=Dropout(dr) (x)

targets = Dense(2, activation="softmax")(x)

model = Model(inputs=inputs, outputs=targets, name="pathfinder_model")


model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

print(model.summary())


class Pathfinder32Generator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_path="data", batch_size=32, dim=( 32, 32, 1),
                 n_classes=2, shuffle=True , n_slice=None, split_mode="easy"):
        'Initialization'
        self.data_path = data_path
        self.batch_size = batch_size
        self.mode = split_mode
        self.dim = dim
        self.shuffle = shuffle
        self.n_slice = n_slice
        self.n_classes = n_classes
        self.split_mode_dict = {
        'easy': "curv_baseline",
        'intermediate': "curv_contour_length_9",
        'hard': "curv_contour_length_14",
        }

        """Read the input data out of the source files."""
        file_path =  os.path.normpath(os.path.join(os.getcwd(),self.data_path ,self.split_mode_dict[split_mode], "metadata")+"/1.npy")

        self.meta_examples = np.load(file_path)
        if self.n_slice != None:
            slice_str = self.n_slice.split(":")
            self.meta_examples = self.meta_examples[int(slice_str[0]):int(slice_str[1])]
            i = 0
            for x in self.meta_examples:
                self.meta_examples[i][2] = i 
                i = i+1

        self.indexes = np.arange(len(self.meta_examples))
        self.on_epoch_end()

    ###should return number of batches per eopch
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.meta_examples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_meta_examples_temp =  indexes
        # Generate data
        X, y = self.__data_generation(list_meta_examples_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.meta_examples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_meta_examples_temp):
        'Generates data containing batch_size samples' 
        # Initialization     
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        idx = 0  
        for index_at in list_meta_examples_temp:

            current_data_image = self.meta_examples[int(index_at)]
            current_image_path = os.path.join(self.data_path ,self.split_mode_dict[self.mode],current_data_image[0], current_data_image[1])
            if self.dim[2] == 3:
                X[idx,] = np.array(tf.keras.preprocessing.image.load_img(current_image_path))
            else:
                X[idx,] = tf.keras.backend.expand_dims(np.array(tf.keras.preprocessing.image.load_img(current_image_path, color_mode = "grayscale")), axis=-1)
            # Store class
            y[idx] = np.array(current_data_image[3])
            idx = idx+1


        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

training_generator = Pathfinder32Generator(n_slice="0:8000",batch_size=batch_size)
test_generator = Pathfinder32Generator(n_slice="8000:10000",batch_size=batch_size)

model.fit_generator(training_generator,
                                 steps_per_epoch=int(8000/batch_size),
                                 epochs=500, 
                                 validation_data = test_generator)
