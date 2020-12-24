from __future__ import print_function


import os
import sys
import cv2


import tensorflow as tf

from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Input, BatchNormalization

sys.path.append(os.path.normpath(os.path.dirname(os.getcwd())))

from mandarin_common_tf2 import *
##############################################################
# File name
##############################################################

mpath = os.path.realpath(__file__)
parent_dir = os.path.dirname(mpath)
BASE_FILE_NAME = os.path.basename(os.path.splitext(mpath)[0])



##############################################################
# Location
##############################################################


##############################################################
# Location
##############################################################

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


##############################################################
# Location
##############################################################

batch_size = 256
num_classes = 2
epochs = 150


##############################################################
# Import dataset from images
##############################################################

VIDS_PATH = os.path.join(parent_dir, 'data')

# easy dataset
easy_folder_PATH = "curv_baseline"
sataset_PATH = os.path.join(easy_folder_PATH,"imgs")
metadata_PATH = os.path.join(easy_folder_PATH,"metadata")

nb_folders = 200
######################################
# finding all images in the root
######################################

x_x = []
y_y = []
labels_dico = {}
Errors_ct = 0

if True:

    for folder_id in range(nb_folders):
        if nb_folders == 1:
            folder_id = folder_id + 1
        if folder_id % 10 == 0:
            print("folder: ", folder_id)

        filename = os.path.join(VIDS_PATH, metadata_PATH, str(folder_id)+".npy")
        print(filename)
        file = open(filename)
        foo = file.readlines()

        for indexo,elem in enumerate(foo):
            res = elem.split(" ")
            labels_dico[res[1]]=res[3]


        VIDS_PATH_perfolder1 = os.path.join(VIDS_PATH, sataset_PATH)
        VIDS_PATH_perfolder = os.path.join(
            VIDS_PATH_perfolder1, str(folder_id))

        KK = [x for x in os.walk(VIDS_PATH_perfolder)]

        X_train = []
        LL = [xx for xx in KK[0][2]]

        if len(LL) != 1000:
            print(len(LL), folder_id)
            sys.exit()

        previous_path = ""
        for indexo, f1 in enumerate(LL):

            new_path = os.path.join(VIDS_PATH_perfolder, f1)

            img = cv2.imread(new_path)

            if img is None:
                print(folder_id, indexo, f1)
                img = cv2.imread(previous_path)
                Errors_ct += 1

            shapeimg = img.shape

            if shapeimg[0] != 32 or shapeimg[1] != 32 or shapeimg[2] != 3:
                print(shapeimg)
                sys.exit()

            x_x.append(img)
            y_y.append(labels_dico[f1])

            previous_path = new_path

    x_x_np = np.asarray(x_x)
    y_y_np = np.asarray(y_y)

    print(x_x_np.shape)

    x_train = x_x[:190000]
    x_test = x_x[190000:]

    y_train = np.asarray(y_y[:190000])
    y_test = np.asarray(y_y[190000:])

    print("y_train.shape", y_train.shape)

    x_train = np.asarray(x_train).astype('float32')
    x_test = np.asarray(x_test).astype('float32')

    x_train /= 255
    x_test /= 255


    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)


print("x_train.shape", x_train.shape)
print("Errors_ct", Errors_ct)


nb_patches = 750  # 750
CONV1D_dim = 24  # 28
l2reg = 0.000001
mDR = 0.5  # 0.1
nb_stacks = 1
transformer_style = True
conv1d_kernel = 1
padding_style = "same"


def get_model():

    input_layer = Input(name='input_layer', shape=(32, 32, 3))

    x = input_layer

    x = Lambda(lambda dd: tf.reshape(dd, [-1, 32*32, 3]))(x)

    x = IGLOO1D_BLOCK(x, nb_patches, CONV1D_dim, patch_size=3, padding_style=padding_style,
              l2_reg=l2reg, nb_stacks=nb_stacks, DR=mDR, conv1d_kernel=conv1d_kernel, pooling_size=1)
    x = BatchNormalization()(x)
    x = Dropout(mDR)(x)

    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(input_layer, output_layer)

    adam = optimizers.Adam(lr=0.0002, clipnorm=5.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    return model


model = get_model()

model.summary()


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[])

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
