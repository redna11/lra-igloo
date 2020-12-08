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


import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Conv1D, Flatten, Conv2D, Embedding, LSTM, LeakyReLU, SpatialDropout1D, Concatenate, Add, Activation, Dropout

from tensorflow.keras.regularizers import l2, l1

import sys
import numpy as np


class MMLinear(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class IGLOO1D_TF2_LAYER:

    def __init__(self, nb_patches, nb_filters_conv1d, patch_size=4, padding_style="causal", nb_stacks=1, conv1d_kernel=3, DR=0.01, l2_reg=0.000001, nb_layers=1):

        self.nb_patches = nb_patches
        self.nb_filters_conv1d = nb_filters_conv1d
        self.patch_size = patch_size
        self.padding_style = padding_style
        self.nb_stacks = nb_stacks
        self.conv1d_kernel = conv1d_kernel
        self.DR = DR
        self.l2_reg = l2_reg
        self.nb_layers = nb_layers
        self.transition_dim = self.nb_patches * self.nb_stacks

        self.addition_layer = Add()

    def __call__(self, _input):

        output = IGLOO1D_BLOCK(_input, self.nb_patches, self.nb_filters_conv1d, patch_size=self.patch_size,
                               padding_style=self.padding_style, nb_stacks=self.nb_stacks, conv1d_kernel=self.conv1d_kernel, DR=self.DR, l2_reg=self.l2_reg)

        if self.nb_layers == 1:
            return output

        else:
            for ii in range(self.nb_layers-1):
                output_tr = Dense(self.transition_dim)(output)
                output_tr = Activation("relu")(output_tr)
                output_tr = Dropout(self.DR)(output_tr)
                output_tr = Dense(self.transition_dim)(output_tr)
                output_tr = Dropout(self.DR)(output_tr)
                output = IGLOO1D_BLOCK(_input, self.nb_patches, self.nb_filters_conv1d, patch_size=self.patch_size,
                                       padding_style=self.padding_style, nb_stacks=self.nb_stacks, conv1d_kernel=self.conv1d_kernel, DR=self.DR, l2_reg=self.l2_reg)
                output = Add()([output, output_tr])

        return output


def IGLOO1D_BLOCK(incoming_layer, nb_patches, nb_filters_conv1d, patch_size=4, padding_style="causal", nb_stacks=1, conv1d_kernel=3, DR=0.01, l2_reg=0.000001, transformer_style=False, spatial_dropout=True, v2=False, pooling_size=1, incoming_proj=0):

    LAYERS = []

    x = Conv1D(nb_filters_conv1d, conv1d_kernel,
               padding=padding_style)(incoming_layer)
    x = LeakyReLU(alpha=0.1)(x)

    if spatial_dropout:
        x = SpatialDropout1D(DR)(x)
    else:
        x = Dropout(DR)(x)

    if not v2:
        x_igloo = IGLOO1D_KERNEL(patch_size, nb_patches, DR, l2_reg,
                                 transformer_style, pooling_size, incoming_proj)(x)
    else:
        x_igloo = IGLOO1D_KERNEL_v2(
            patch_size, nb_patches, DR, l2_reg, transformer_style, pooling_size)(x)

    LAYERS.append(x_igloo)

    if nb_stacks > 1:

        for extra_l in range(nb_stacks-1):

            x = Conv1D(nb_filters_conv1d, conv1d_kernel,
                       padding=padding_style)(x)
            x = LeakyReLU(alpha=0.1)(x)

            if spatial_dropout:
                x = SpatialDropout1D(DR)(x)
            else:
                x = Dropout(DR)(x)

            if not v2:
                LAYERS.append(IGLOO1D_KERNEL(patch_size, nb_patches, DR,
                                             l2_reg, transformer_style, pooling_size, incoming_proj)(x))
            else:
                LAYERS.append(IGLOO1D_KERNEL_v2(
                    patch_size, nb_patches, DR, l2_reg, transformer_style, pooling_size)(x))

    if nb_stacks > 1:
        MPI = Concatenate()(LAYERS)
    else:
        MPI = LAYERS[0]

    return MPI


class IGLOO1D_KERNEL_v2(tf.keras.layers.Layer):
    def __init__(self, patch_size, nb_patches, DR, l2_reg, transformer_style):
        super(IGLOO1D_KERNEL_v2, self).__init__()

        self.patch_size = patch_size
        self.nb_patches = nb_patches
        self.DR = DR
        self.l2_reg = l2_reg
        self.transformer_style = transformer_style

    def PATCHY_LAYER_CNNTOP_LAST_initializer(self, shape, dtype=None):

        M = gen_filters_igloo_newstyle1Donly(self.patch_size, self.nb_patches, self.vector_size,
                                             self.num_channels_input, build_backbone=False, return_sequences=False)
        M.astype(int)

        return M

    def build(self, input_shape):

        self.batch_size = input_shape[0]
        self.vector_size = input_shape[1]
        self.num_channels_input = input_shape[2]

        # SETTING THIS AS A NON TRAINABLE VARIABLE SO IT CAN BE SAVED ALONG THE WEIGHTS
        self.patches = self.add_weight(shape=(int(self.nb_patches), self.patch_size, 1),
                                       initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializer,
                                       trainable=False,
                                       name="random_patches", dtype=np.int32)
        self.W_SUMMER = self.add_weight(shape=(1, self.patch_size*self.num_channels_input, 1),
                                        initializer="glorot_uniform",
                                        trainable=True,
                                        regularizer=l2(self.l2_reg),
                                        name="W_SUMMER")
        self.W_Q = self.add_weight(shape=(1, self.num_channels_input, self.num_channels_input),
                                   initializer="glorot_uniform",
                                   trainable=True,
                                   regularizer=l2(self.l2_reg),
                                   name="W_V")
        if self.transformer_style:
            self.W_QK = self.add_weight(shape=(self.nb_patches, self.vector_size/4),
                                        initializer="glorot_uniform",
                                        trainable=True,
                                        regularizer=l2(self.l2_reg),
                                        name="W_QK")
            self.W_V = self.add_weight(shape=(1, self.num_channels_input, self.num_channels_input),
                                       initializer="glorot_uniform",
                                       trainable=True,
                                       regularizer=l2(self.l2_reg),
                                       name="W_V")

    def call(self, y):

        sys.exit()

        M = tf.matmul(y, self.W_Q)
        M = tf.transpose(M, [1, 2, 0])
        M = tf.gather_nd(M, self.patches)
        MPI = tf.transpose(M, [3, 0, 1, 2])
        MPI = tf.reshape(
            MPI, [-1, self.nb_patches, self.patch_size*self.num_channels_input])
        MPI = tf.matmul(MPI, self.W_SUMMER)
        MPI = tf.squeeze(MPI, axis=-1)
        sys.exit()

        if self.transformer_style:

            y_proj = tf.matmul(y, self.W_V)
            y_proj = tf.keras.layers.MaxPool1D(pool_size=4)(y_proj)
            alpha = tf.matmul(MPI, self.W_QK)
            alpha = tf.nn.softmax(alpha)
            MPI = tf.matmul(tf.expand_dims(alpha, axis=1), y_proj)
            MPI = tf.squeeze(MPI, axis=1)
        else:
            MPI = LeakyReLU(alpha=0.1)(MPI)

        return MPI


class IGLOO1D_KERNEL(tf.keras.layers.Layer):
    def __init__(self, patch_size, nb_patches, DR, l2_reg, transformer_style, pooling_size, incoming_proj):
        super(IGLOO1D_KERNEL, self).__init__()

        self.patch_size = patch_size
        self.nb_patches = nb_patches
        self.DR = DR
        self.l2_reg = l2_reg
        self.pooling_size = pooling_size
        self.incoming_proj_dim = incoming_proj
        self.transformer_style = transformer_style

    def PATCHY_LAYER_CNNTOP_LAST_initializer(self, shape, dtype=None):

        M = gen_filters_igloo_newstyle1Donly(self.patch_size, self.nb_patches, self.vector_size,
                                             self.num_channels_input, build_backbone=False, return_sequences=False)
        M.astype(int)

        return M

    def build(self, input_shape):

        self.batch_size = input_shape[0]
        self.vector_size = input_shape[1]

        if self.incoming_proj_dim > 0:
            self.W_incoming = self.add_weight(shape=(input_shape[2], self.incoming_proj_dim),
                                              initializer="glorot_uniform",
                                              trainable=True,
                                              regularizer=l2(self.l2_reg),
                                              name="W_incoming")
            self.num_channels_input = self.incoming_proj_dim
        else:
            self.num_channels_input = input_shape[2]

        # SETTING THIS AS A NON TRAINABLE VARIABLE SO IT CAN BE SAVED ALONG THE WEIGHTS
        self.patches = self.add_weight(shape=(int(self.nb_patches), self.patch_size, 1),
                                       initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializer,
                                       trainable=False,
                                       name="random_patches", dtype=np.int32)
        self.W_MULT = self.add_weight(shape=(1, self.nb_patches, self.patch_size, self.num_channels_input),
                                      initializer="glorot_uniform",
                                      trainable=True,
                                      regularizer=l2(self.l2_reg),
                                      name="W_MULT")
        self.W_BIAS = self.add_weight(shape=(1, self.nb_patches),
                                      initializer="glorot_uniform",
                                      trainable=True,
                                      regularizer=l2(self.l2_reg),
                                      name="W_BIAS")
        self.W_SUMMER = self.add_weight(shape=(1, self.patch_size*self.num_channels_input, 1),
                                        initializer="glorot_uniform",
                                        trainable=True,
                                        regularizer=l2(self.l2_reg),
                                        name="W_SUMMER")
        if self.transformer_style:
            self.W_QK = self.add_weight(shape=(self.nb_patches, int(self.vector_size/self.pooling_size)),
                                        initializer="glorot_uniform",
                                        trainable=True,
                                        regularizer=l2(self.l2_reg),
                                        name="W_QK")
            self.W_V = self.add_weight(shape=(1, self.num_channels_input, self.num_channels_input),
                                       initializer="glorot_uniform",
                                       trainable=True,
                                       regularizer=l2(self.l2_reg),
                                       name="W_V")

    def call(self, y):
        if self.incoming_proj_dim > 0:
            y_next = tf.matmul(y, self.W_incoming)
        else:
            y_next = y
        M = tf.transpose(y_next, [1, 2, 0])
        M = tf.gather_nd(M, self.patches)
        MPI = tf.transpose(M, [3, 0, 1, 2])
        MPI = tf.multiply(self.W_MULT, MPI)
        MPI = tf.reshape(
            MPI, [-1, self.nb_patches, self.patch_size*self.num_channels_input])
        MPI = tf.matmul(MPI, self.W_SUMMER)
        MPI = tf.squeeze(MPI, axis=-1)
        MPI = MPI + self.W_BIAS
        if self.transformer_style:
            y_proj = tf.matmul(y, self.W_V)
            if self.pooling_size > 1:
                y_proj = tf.keras.layers.MaxPool1D(
                    pool_size=self.pooling_size)(y_proj)
            alpha = tf.matmul(MPI, self.W_QK)
            alpha = tf.nn.softmax(alpha)
            MPI = tf.matmul(tf.expand_dims(alpha, axis=1), y_proj)
            MPI = tf.squeeze(MPI, axis=1)
        else:
            MPI = LeakyReLU(alpha=0.1)(MPI)

        return MPI


#################################################################################################
# make the patches
#################################################################################################


def gen_filters_igloo_newstyle1Donly(patch_size, nb_patches, vector_size, num_channels_input_reduced, return_sequences, nb_stacks=1, build_backbone=False, consecutive=False, nb_sequences=-1):

    OUTA = []
    vector_size = int(vector_size)
    if nb_stacks == 1:
        for step in range(vector_size):
            if (step != vector_size-1) and (return_sequences == False):
                continue
            if return_sequences == True and (nb_sequences != -1):
                if step < vector_size-nb_sequences:
                    continue

            if step % 10 == 0:

            COLLECTA = []

            #######################
            # if small step ID
            #######################
            if step < patch_size:
                for kk in range(nb_patches):
                    randy_H = np.random.choice(
                        range(step+1), patch_size, replace=True)
                    first = []
                    for pp in randy_H:
                        first.append([pp])
                    COLLECTA.append(first)

                OUTA.append(COLLECTA)
            #######################
            # if larger step ID
            #######################
            else:
                # first manufactur the arranged ones :
                #################
                # building backbone
                #################
                if build_backbone:
                    maximum_its = int((step/(patch_size-1))+1)
                    if maximum_its > nb_patches:
                        print("nb_patches too small, recommende above:", maximum_its)
                        sys.exit()
                    for jj in range(maximum_its):
                        if iter == 0:
                            randy_H = [step-pp for pp in range(patch_size)]
                        else:
                            randy_H = [max(step-(jj*(patch_size-1))-pp, 0)
                                       for pp in range(patch_size)]
                        first = []
                        for pp in randy_H:
                            first.append([pp])
                        COLLECTA.append(first)

                    ###########################################
                    # doing rest of iterations as freestyle
                    ###########################################
                    rest_iters = max(nb_patches-maximum_its, 0)

                    for itero in range(rest_iters):
                        if not consecutive:
                            randy_B = np.random.choice(
                                range(step+1), patch_size, replace=False)
                        else:
                            uniq = np.random.choice(
                                range(max(0, step+1-patch_size+1)), 1, replace=False)
                            randy_B = [uniq[0]+pp for pp in range(patch_size)]
                        first = []
                        sorting = True  # set to true , almost 15pc faster
                        if sorting:
                            randy_B = sorted(randy_B)
                        for pp in randy_B:
                            first.append([pp])
                        COLLECTA.append(first)
                    COLLECTA = np.stack(COLLECTA)
                    OUTA.append(COLLECTA)

                #################
                # NOT building backbone
                #################
                else:
                    for itero in range(nb_patches):
                        if not consecutive:
                            randy_B = np.random.choice(
                                range(step+1), patch_size, replace=False)
                        else:
                            uniq = np.random.choice(
                                range(max(0, step+1-patch_size+1)), 1, replace=False)
                            randy_B = [uniq[0]+pp for pp in range(patch_size)]
                        first = []
                        sorting = True
                        if sorting:
                            randy_B = sorted(randy_B)
                        for pp in randy_B:
                            first.append([pp])
                        COLLECTA.append(first)
                    COLLECTA = np.stack(COLLECTA)
                    OUTA.append(COLLECTA)
        OUTA = np.stack(OUTA)
        if return_sequences == False:
            OUTA = np.squeeze(OUTA, axis=0)
        return OUTA

    else:
        sys.exit()
        for step in range(vector_size):
            if (step != vector_size-1) and (return_sequences == False):
                continue

            if return_sequences == True and (nb_sequences != -1):
                if step < vector_size-nb_sequences:
                    continue

            if step % 10 == 0:
                print("step...", step)

            COLLECTA = []

            if step < patch_size:
                print("The part for multi stack in ONE SHOT has not been dev yet")
                sys.exit()
                for kk in range(nb_patches):
                    randy_H = np.random.choice(
                        range(step+1), patch_size, replace=True)
                    first = []
                    for pp in randy_H:
                        for mm in range(num_channels_input_reduced):
                            first.append([pp, mm])
                    COLLECTA.append(first)
                OUTA.append(COLLECTA)

            else:
                if build_backbone:
                    maximum_its = int((step/(patch_size-1))+1)
                    if maximum_its > nb_patches:
                        print("nb_patches too small, recommende above:", maximum_its)
                        sys.exit()

                    for jj in range(maximum_its):
                        if iter == 0:
                            randy_H = [step-pp for pp in range(patch_size)]
                        else:
                            randy_H = [max(step-(jj*(patch_size-1))-pp, 0)
                                       for pp in range(patch_size)]

                        first = []
                        for pp in randy_H:
                            for mm in range(num_channels_input_reduced):
                                first.append([pp, mm])

                        COLLECTA.append(first)
                    rest_iters = max(nb_patches-maximum_its, 0)
                    for itero in range(rest_iters):
                        if not consecutive:
                            randy_B = np.random.choice(
                                range(step+1), patch_size, replace=False)
                        else:
                            uniq = np.random.choice(
                                range(max(0, step+1-patch_size+1)), 1, replace=False)
                            randy_B = [uniq[0]+pp for pp in range(patch_size)]
                        first = []
                        for pp in randy_B:
                            for mm in range(num_channels_input_reduced):
                                first.append([pp, mm])
                        COLLECTA.append(first)
                    COLLECTA = np.stack(COLLECTA)
                    OUTA.append(COLLECTA)

                ###################################
                # if not building backbone
                ##################################

                else:
                    sys.exit()
                    for stack_id in range(nb_stacks):
                        for itero in range(nb_patches):
                            if not consecutive:
                                randy_B = np.random.choice(
                                    range(step+1), patch_size, replace=False)
                            else:
                                uniq = np.random.choice(
                                    range(max(0, step+1-patch_size+1)), 1, replace=False)
                                randy_B = [
                                    uniq[0]+pp for pp in range(patch_size)]
                            first = []
                            sys.exit()
                            for pp in randy_B:
                                for mm in range(num_channels_input_reduced):
                                    first.append([stack_id, pp, mm])
                            COLLECTA.append(first)
                    COLLECTA = np.stack(COLLECTA)
                    OUTA.append(COLLECTA)
        OUTA = np.stack(OUTA)
        if return_sequences == False:
            OUTA = np.squeeze(OUTA, axis=0)
        return OUTA
