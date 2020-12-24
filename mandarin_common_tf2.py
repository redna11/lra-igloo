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

import tensorflow.keras.backend as K

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



def cos_distance_anyD_pairwise(a, b, remove_naan=False):

    norma = tf.expand_dims(tf.sqrt(tf.reduce_sum(
        tf.square(a)+0.0000000001, -1)), axis=-1)
    normb = tf.expand_dims(tf.sqrt(tf.reduce_sum(
        tf.square(b)+0.0000000001, -1)), axis=-1)
    a = a/norma
    b = b/normb

    QQ = tf.multiply(a, b)

    QQ = tf.reduce_sum(QQ, axis=-1)

    if remove_naan:
        QQ = tf.where(tf.is_nan(QQ), 0.00001*tf.zeros_like(QQ), QQ)

    return QQ


def IGLOO1D_BLOCK(incoming_layer, nb_patches, nb_filters_conv1d, patch_size=4, padding_style="causal", nb_stacks=1, conv1d_kernel=3, DR=0.01, l2_reg=0.000001, transformer_style=False, spatial_dropout=True, blockstyle="v1", pooling_size=1, incoming_proj=0):

    LAYERS = []

    x = Conv1D(nb_filters_conv1d, conv1d_kernel,
               padding=padding_style)(incoming_layer)
    x = LeakyReLU(alpha=0.1)(x)

    if spatial_dropout:
        x = SpatialDropout1D(DR)(x)
    else:
        x = Dropout(DR)(x)

    if blockstyle == "v1":
        x_igloo = IGLOO1D_KERNEL(patch_size, nb_patches, DR, l2_reg,
                                 transformer_style, pooling_size, incoming_proj)(x)

    elif blockstyle == "v2":
        x_igloo = IGLOO1D_KERNEL_v2(
            patch_size, nb_patches, DR, l2_reg, transformer_style, pooling_size, incoming_proj)(x)

    elif blockstyle == "v3":
        x_igloo = IGLOO1D_KERNEL_v3(
            patch_size, nb_patches, DR, l2_reg, transformer_style, pooling_size, incoming_proj)(x)

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

        if blockstyle == "v1":
            x_igloo = IGLOO1D_KERNEL(
                patch_size, nb_patches, DR, l2_reg, transformer_style, pooling_size, incoming_proj)(x)

        elif blockstyle == "v2":
            x_igloo = IGLOO1D_KERNEL_v2(
                patch_size, nb_patches, DR, l2_reg, transformer_style, pooling_size, incoming_proj)(x)

        elif blockstyle == "v3":
            x_igloo = IGLOO1D_KERNEL_v3(
                patch_size, nb_patches, DR, l2_reg, transformer_style, pooling_size, incoming_proj)(x)

        LAYERS.append(x_igloo)

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

        print("not in use")
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

            self.num_channels_input = input_shape[2]

        else:
            self.num_channels_input = input_shape[2]

        self.patches = self.add_weight(shape=(int(self.nb_patches), self.patch_size, 1),
                                       initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializer,
                                       trainable=False,
                                       name="random_patches", dtype=np.int32)

        if self.incoming_proj_dim == 0:
            self.W_MULT = self.add_weight(shape=(1, self.nb_patches, self.patch_size, self.num_channels_input),
                                          initializer="glorot_uniform",
                                          trainable=True,
                                          regularizer=l2(self.l2_reg),
                                          name="W_MULT")

            self.W_SUMMER = self.add_weight(shape=(1, self.patch_size*self.num_channels_input, 1),
                                            initializer="glorot_uniform",
                                            trainable=True,
                                            regularizer=l2(self.l2_reg),
                                            name="W_SUMMER")

        else:
            self.W_MULT = self.add_weight(shape=(1, self.nb_patches, self.patch_size, self.incoming_proj_dim),
                                          initializer="glorot_uniform",
                                          trainable=True,
                                          regularizer=l2(self.l2_reg),
                                          name="W_MULT")

            self.W_SUMMER = self.add_weight(shape=(1, self.patch_size*self.incoming_proj_dim, 1),
                                            initializer="glorot_uniform",
                                            trainable=True,
                                            regularizer=l2(self.l2_reg),
                                            name="W_SUMMER")

        self.W_BIAS = self.add_weight(shape=(1, self.nb_patches),
                                      initializer="glorot_uniform",
                                      trainable=True,
                                      regularizer=l2(self.l2_reg),
                                      name="W_BIAS")


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
        if self.incoming_proj_dim == 0:
            MPI = tf.reshape(
                MPI, [-1, self.nb_patches, self.patch_size*self.num_channels_input])
        else:
            MPI = tf.reshape(
                MPI, [-1, self.nb_patches, self.patch_size*self.incoming_proj_dim])

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




class IGLOO1D_KERNEL_v3(tf.keras.layers.Layer):
    def __init__(self, patch_size, nb_patches, DR, l2_reg, transformer_style, pooling_size, incoming_proj):
        super(IGLOO1D_KERNEL_v3, self).__init__()

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

            self.num_channels_input = input_shape[2]

        else:
            self.num_channels_input = input_shape[2]

        self.patches = self.add_weight(shape=(int(self.nb_patches), self.patch_size, 1),
                                       initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializer,
                                       trainable=False,
                                       name="random_patches", dtype=np.int32)

        self.patches2 = self.add_weight(shape=(int(self.nb_patches), self.patch_size, 1),
                                        initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializer,
                                        trainable=False,
                                        name="random_patches2", dtype=np.int32)

        if self.incoming_proj_dim == 0:
            self.W_MULT = self.add_weight(shape=(1, self.nb_patches, self.patch_size, self.num_channels_input),
                                          initializer="glorot_uniform",
                                          trainable=True,
                                          regularizer=l2(self.l2_reg),
                                          name="W_MULT")

            self.W_SUMMER = self.add_weight(shape=(1, self.patch_size*self.num_channels_input, 1),
                                            initializer="glorot_uniform",
                                            trainable=True,
                                            regularizer=l2(self.l2_reg),
                                            name="W_SUMMER")

        else:
            self.W_MULT = self.add_weight(shape=(1, self.nb_patches, self.patch_size, self.incoming_proj_dim),
                                          initializer="glorot_uniform",
                                          trainable=True,
                                          regularizer=l2(self.l2_reg),
                                          name="W_MULT")

            self.W_SUMMER = self.add_weight(shape=(1, self.patch_size*self.incoming_proj_dim, 1),
                                            initializer="glorot_uniform",
                                            trainable=True,
                                            regularizer=l2(self.l2_reg),
                                            name="W_SUMMER")

        self.W_BIAS = self.add_weight(shape=(1, self.nb_patches),
                                      initializer="glorot_uniform",
                                      trainable=True,
                                      regularizer=l2(self.l2_reg),
                                      name="W_BIAS")

        if self.transformer_style:

            self.W_QK = self.add_weight(shape=(self.nb_patches, int(self.nb_patches/self.pooling_size)),
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
        if self.incoming_proj_dim == 0:
            MPI = tf.reshape(
                MPI, [-1, self.nb_patches, self.patch_size*self.num_channels_input])
        else:
            MPI = tf.reshape(
                MPI, [-1, self.nb_patches, self.patch_size*self.incoming_proj_dim])

        MPI = tf.matmul(MPI, self.W_SUMMER)
        MPI = tf.squeeze(MPI, axis=-1)
        MPI = MPI + self.W_BIAS

        if self.transformer_style:

            y_proj = tf.matmul(y, self.W_V)

            y_proj = tf.transpose(y_proj, [1, 2, 0])
            y_proj = tf.gather_nd(y_proj, self.patches2)
            y_proj = tf.transpose(y_proj, [3, 0, 1, 2]) 

            y_proj = tf.reduce_sum(y_proj, axis=2)  

            alpha = tf.matmul(MPI, self.W_QK)
            alpha = tf.nn.softmax(alpha)
            MPI = tf.matmul(tf.expand_dims(alpha, axis=1), y_proj)
            MPI = tf.squeeze(MPI, axis=1)

        else:

            MPI = LeakyReLU(alpha=0.1)(MPI)


        return MPI


class IGLOO1D_KERNEL_pat(tf.keras.layers.Layer):
    def __init__(self, patch_size, nb_patches, DR, l2_reg, transformer_style, pooling_size, incoming_proj):
        super(IGLOO1D_KERNEL_pat, self).__init__()

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

    def PATCHY_LAYER_CNNTOP_LAST_initializerK(self, shape, dtype=None):

        M = gen_filters_igloo_newstyle1Donly(self.patch_size, self.nb_patches, self.vector_size,
                                             self.num_channels_input, build_backbone=False, return_sequences=False)
        M.astype(int)

        return M

    def build(self, input_shape):

        self.batch_size = input_shape[0]
        self.vector_size = input_shape[1]

        if self.incoming_proj_dim > 0:

            self.W_incoming_Q = self.add_weight(shape=(input_shape[2], self.incoming_proj_dim),
                                                initializer="glorot_uniform",
                                                trainable=True,
                                                regularizer=l2(self.l2_reg),
                                                name="W_incoming_Q")

            self.W_incoming_K = self.add_weight(shape=(input_shape[2], self.incoming_proj_dim),
                                                initializer="glorot_uniform",
                                                trainable=True,
                                                regularizer=l2(self.l2_reg),
                                                name="W_incoming_K")

            self.num_channels_input = input_shape[2]

        else:
            self.num_channels_input = input_shape[2]


        self.patches_Q = self.add_weight(shape=(int(self.nb_patches), self.patch_size, 1),
                                         initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializer,
                                         trainable=False,
                                         name="random_patches_Q", dtype=np.int32)

        self.patches_K = self.add_weight(shape=(int(self.nb_patches), self.patch_size, 1),
                                         initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializerK,
                                         trainable=False,
                                         name="random_patches_K", dtype=np.int32)

        self.W_MULT = self.add_weight(shape=(1, self.nb_patches, self.patch_size, self.incoming_proj_dim),
                                      initializer="glorot_uniform",
                                      trainable=True,
                                      regularizer=l2(self.l2_reg),
                                      name="W_MULT")

        self.W_SUMMER = self.add_weight(shape=(1, self.patch_size*self.incoming_proj_dim, 1),
                                        initializer="glorot_uniform",
                                        trainable=True,
                                        regularizer=l2(self.l2_reg),
                                        name="W_SUMMER")

        self.W_BIAS = self.add_weight(shape=(1, self.nb_patches),
                                      initializer="glorot_uniform",
                                      trainable=True,
                                      regularizer=l2(self.l2_reg),
                                      name="W_BIAS")

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
            y_next_Q = tf.matmul(y, self.W_incoming_Q)
            y_next_K = tf.matmul(y, self.W_incoming_K)
        else:
  
            sys.exit()
            y_next_Q = y

        M_Q = tf.transpose(y_next_Q, [1, 2, 0])
        M_Q = tf.gather_nd(M_Q, self.patches_Q)
        MPI_Q = tf.transpose(M_Q, [3, 0, 1, 2])

        M_K = tf.transpose(y_next_K, [1, 2, 0])
        M_K = tf.gather_nd(M_K, self.patches_K)
        MPI_K = tf.transpose(M_K, [3, 0, 1, 2])

        MPI_Q = tf.multiply(MPI_Q, self.W_MULT)

        if self.incoming_proj_dim > 0:
            MPI_Q = tf.reshape(
                MPI_Q, [-1, self.nb_patches, self.patch_size*self.num_channels_input])
            MPI_K = tf.reshape(
                MPI_K, [-1, self.nb_patches, self.patch_size*self.num_channels_input])
        else:
    
            sys.exit()
            MPI = tf.reshape(
                MPI, [-1, self.nb_patches, self.patch_size*self.incoming_proj_dim])



        MPI = tf.multiply(MPI_Q, MPI_K)

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


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(
            K.square(inputs - mean), axis=self.axis, keepdims=True)
        epsilon = K.constant(1e-5, dtype=K.floatx())
        normalized_inputs = (inputs - mean) / K.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class BB_Block_v1():

    def __init__(self, name):

        self.residual_dropout = 0.5
        self.mDR_local = 0.5

        self.norm1_layer = LayerNormalization(name=f'{name}_normalization1')
        self.norm2_layer = LayerNormalization(name=f'{name}_normalization2')
        self.addition_layer = Add(name=f'{name}_add')
        self.nb_patches = 1000

        self.nb_filters_conv1d = 256
        self.patch_size = 650
        self.nb_stacks = 1
        self.l2_reg = 0

        self.spatial_dropout = False
        self.padding_style = "same"
        self.conv1d_kernel = 1

    def __call__(self, _input):

        LARGE_MODEL = True

        output = IGLOO_BB_v1(_input)

        MPI = Add()([_input, output])

        norm1_output = self.norm1_layer(MPI)

        output = Dense(512)(norm1_output)
        output = Activation("relu")(output)
        output = Dropout(self.mDR_local)(output)
        output = Dense(256)(output)

        output = Dropout(self.mDR_local)(output)

        QQ = Add()([norm1_output, output])

        QQ = self.norm2_layer(QQ)

        return QQ


def gen_filters_igloo_newstyle1Donly(patch_size, nb_patches, vector_size, num_channels_input_reduced, return_sequences, nb_stacks=1, build_backbone=True, consecutive=False, nb_sequences=-1):

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
                print("step...", step)

            COLLECTA = []

            if step < patch_size:

                for kk in range(nb_patches):

                    randy_H = np.random.choice(
                        range(step+1), patch_size, replace=True)

                    first = []

                    for pp in randy_H:
                        first.append([pp])

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
                            first.append([pp])

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

                        sorting = True 
                        if sorting:
                            randy_B = sorted(randy_B)

                        for pp in randy_B:
                            first.append([pp])


                        COLLECTA.append(first)

                    COLLECTA = np.stack(COLLECTA)
                    OUTA.append(COLLECTA)

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

        print("need to dev more")
        sys.exit()

        for step in range(vector_size):  

            if (step != vector_size-1) and (return_sequences == False):
                continue

            if return_sequences == True and (nb_sequences != -1):
                if step < vector_size-nb_sequences:
                    continue


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


def IGLOO_BB_v1(incoming_layer):

    nb_patches = 100
    nb_filters_conv1d = 256
    patch_size = 3
    padding_style = "same"
    nb_stacks = 1
    conv1d_kernel = 1,
    DR = 0.01
    l2_reg = 0.000001
    spatial_dropout = True

    LAYERS = []

    x = Conv1D(nb_filters_conv1d, conv1d_kernel,
               padding=padding_style)(incoming_layer)
    x = LeakyReLU(alpha=0.1)(x)

    if spatial_dropout:
        x = SpatialDropout1D(DR)(x)
    else:
        x = Dropout(DR)(x)

    x_igloo = IGLOO1Dseq_KERNEL(patch_size, nb_patches, DR, l2_reg)(x)

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

        x_igloo = IGLOO1Dseq_KERNEL(patch_size, nb_patches, DR, l2_reg)(x)

    if nb_stacks > 1:

        MPI = Concatenate(axis=-1)(LAYERS)

        x = Conv1D(256, 1, padding="same")(reso)

    else:

        MPI = LAYERS[0]

    return MPI


class IGLOO1Dseq_KERNEL_WK2style(tf.keras.layers.Layer):
    def __init__(self, patch_size, nb_patches, DR, l2_reg):
        super(IGLOO1Dseq_KERNEL, self).__init__()

        self.patch_size = patch_size
        self.nb_patches = nb_patches
        self.DR = DR
        self.l2_reg = l2_reg
        self.patch_size_proj = 1
        self.mb_size = 128
        self.compression_size = 5
        self.final_dim = 128

    def PATCHY_LAYER_CNNTOP_LAST_initializer(self, shape, dtype=None):


        M = gen_seq_classify(0.75, self.patch_size, self.nb_patches, int(self.vector_size), int(1*self.compression_size),
                             steps_limit=0, return_sequences=True, stretch_factor=1, nb_sequences=-1, build_backbone=False, last_only=False)
        M.astype(int)

        return M

    def PATCHY_LAYER_CNNTOP_LAST_initializer2(self, shape, dtype=None):


        M = gen_seq_classify(0.75, self.patch_size_proj, self.nb_patches, int(self.vector_size), int(
            1*self.compression_size), steps_limit=0, return_sequences=True, stretch_factor=1, nb_sequences=-1, build_backbone=False, last_only=False)
        M.astype(int)

        return M

    def build(self, input_shape):

        self.batch_size = input_shape[0]
        self.vector_size = input_shape[1]
        self.num_channels_input = input_shape[2]


        self.patches = self.add_weight(shape=(self.vector_size, int(self.nb_patches), self.patch_size, 1),
                                       initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializer,
                                       trainable=False,
                                       name="random_patches", dtype=np.int32)


        self.patches_proj = self.add_weight(shape=(self.vector_size, int(self.nb_patches), self.patch_size_proj, 1),
                                            initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializer2,
                                            trainable=False,
                                            name="patches_proj", dtype=np.int32)

        self.W_BASE = self.add_weight(shape=(1, 1, self.nb_patches, self.patch_size, self.compression_size),
                                      initializer="glorot_uniform",
                                      trainable=True,
                                      regularizer=l2(self.l2_reg),
                                      name="W_BASE")

        self.W_BIAS = self.add_weight(shape=(1, 1, self.nb_patches),
                                      initializer="glorot_uniform",
                                      trainable=True,
                                      regularizer=l2(self.l2_reg),
                                      name="W_BIAS")

        self.W_proj2 = self.add_weight(shape=(1, self.num_channels_input, self.mb_size),
                                       initializer="glorot_uniform",
                                       trainable=True,
                                       regularizer=l2(self.l2_reg),
                                       name="W_proj2")

        self.W_proj_to_compression = self.add_weight(shape=(1, self.num_channels_input, self.compression_size),
                                                     initializer="glorot_uniform",
                                                     trainable=True,
                                                     regularizer=l2(
            self.l2_reg),
            name="W_proj_to_compression")

        self.W_presoftmax1 = self.add_weight(shape=(1, self.nb_patches, self.nb_patches),
                                             initializer="glorot_uniform",
                                             trainable=True,
                                             regularizer=l2(self.l2_reg),
                                             name="W_presoftmax1")

        self.W_FINAL = self.add_weight(shape=(1, self.mb_size, self.final_dim),
                                       initializer="glorot_uniform",
                                       trainable=True,
                                       regularizer=l2(self.l2_reg),
                                       name="W_FINAL")

    def call(self, y):


        BM_projected = tf.matmul(y, self.W_proj2)
        BM_projected = tf.transpose(BM_projected, [1, 2, 0])
        BM_projected = tf.gather_nd(BM_projected, self.patches_proj)
        BM_projected = tf.squeeze(BM_projected, axis=2)
        MBMB = tf.transpose(BM_projected, [3, 0, 1, 2])

        y_compressed = tf.matmul(y, self.W_proj_to_compression)
        Q = tf.transpose(y_compressed, [1, 2, 0])
        Q = tf.gather_nd(Q, self.patches)

        Q = tf.transpose(Q, [4, 0, 1, 2, 3])
        Q = Q*self.W_BASE
        RES = tf.reduce_sum(Q, axis=[3, 4])
        RES = tf.add(RES, self.W_BIAS)
        RES = tf.matmul(RES, self.W_presoftmax1)


        RES = tf.expand_dims(RES, axis=2)
        RES = tf.matmul(RES, MBMB)
        RES = tf.squeeze(RES, axis=2)
        RES = tf.matmul(RES, self.W_FINAL)

        return RES


class IGLOO1Dseq_KERNEL(tf.keras.layers.Layer):
    def __init__(self, patch_size, nb_patches, DR, l2_reg):
        super(IGLOO1Dseq_KERNEL, self).__init__()

        self.patch_size = patch_size
        self.nb_patches = nb_patches
        self.DR = DR
        self.l2_reg = l2_reg
        self.patch_size_proj = 1
        self.mb_size = 256
        self.compression_size = 5
        self.final_dim = 256

    def PATCHY_LAYER_CNNTOP_LAST_initializer(self, shape, dtype=None):

        M = gen_seq_classify(0.75, self.patch_size, self.nb_patches, int(self.vector_size), int(1*self.compression_size),
                             steps_limit=0, return_sequences=True, stretch_factor=1, nb_sequences=-1, build_backbone=False, last_only=False)
        M.astype(int)

        return M

    def PATCHY_LAYER_CNNTOP_LAST_initializer2(self, shape, dtype=None):

        M = gen_seq_classify(0.75, self.patch_size_proj, self.nb_patches, int(self.vector_size), int(
            1*self.compression_size), steps_limit=0, return_sequences=True, stretch_factor=1, nb_sequences=-1, build_backbone=False, last_only=False)
        M.astype(int)


        return M

    def build(self, input_shape):

        self.batch_size = input_shape[0]
        self.vector_size = input_shape[1]
        self.num_channels_input = input_shape[2]


        self.patches = self.add_weight(shape=(self.vector_size, int(self.nb_patches), self.patch_size, 1),
                                       initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializer,
                                       trainable=False,
                                       name="random_patches", dtype=np.int32)

        self.patches_proj = self.add_weight(shape=(self.vector_size, int(self.nb_patches), self.patch_size_proj, 1),
                                            initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializer2,
                                            trainable=False,
                                            name="patches_proj", dtype=np.int32)

        self.W_BASE = self.add_weight(shape=(1, 1, self.nb_patches, self.patch_size, self.compression_size),
                                      initializer="glorot_uniform",
                                      trainable=True,
                                      regularizer=l2(self.l2_reg),
                                      name="W_BASE")

        self.W_BIAS = self.add_weight(shape=(1, 1, self.nb_patches),
                                      initializer="glorot_uniform",
                                      trainable=True,
                                      regularizer=l2(self.l2_reg),
                                      name="W_BIAS")

        self.W_proj2 = self.add_weight(shape=(1, self.num_channels_input, self.mb_size),
                                       initializer="glorot_uniform",
                                       trainable=True,
                                       regularizer=l2(self.l2_reg),
                                       name="W_proj2")

        self.W_proj_to_compression = self.add_weight(shape=(1, self.num_channels_input, self.compression_size),
                                                     initializer="glorot_uniform",
                                                     trainable=True,
                                                     regularizer=l2(
            self.l2_reg),
            name="W_proj_to_compression")

        self.W_presoftmax1 = self.add_weight(shape=(1, self.nb_patches, self.nb_patches),
                                             initializer="glorot_uniform",
                                             trainable=True,
                                             regularizer=l2(self.l2_reg),
                                             name="W_presoftmax1")

        self.W_FINAL = self.add_weight(shape=(1, self.mb_size, self.final_dim),
                                       initializer="glorot_uniform",
                                       trainable=True,
                                       regularizer=l2(self.l2_reg),
                                       name="W_FINAL")

    def call(self, y):

        BM_projected = tf.matmul(y, self.W_proj2)
        BM_projected = tf.transpose(BM_projected, [1, 2, 0])
        BM_projected = tf.gather_nd(BM_projected, self.patches_proj)
        BM_projected = tf.squeeze(BM_projected, axis=2)
        MBMB = tf.transpose(BM_projected, [3, 0, 1, 2])

        y_compressed = tf.matmul(y, self.W_proj_to_compression)
        Q = tf.transpose(y_compressed, [1, 2, 0])

        Q = tf.gather_nd(Q, self.patches)  
        Q = tf.transpose(Q, [4, 0, 1, 2, 3])
        Q = Q*self.W_BASE

        RES = tf.reduce_sum(Q, axis=[3, 4])
        RES = tf.add(RES, self.W_BIAS)
        RES = tf.matmul(RES, self.W_presoftmax1)

        RES = tf.expand_dims(RES, axis=2)
        RES = tf.matmul(RES, MBMB)
        RES = tf.squeeze(RES, axis=2)
        RES = tf.matmul(RES, self.W_FINAL)

        return RES


class ADDCLS(tf.keras.layers.Layer):
    def __init__(self, l2_reg):
        super(ADDCLS, self).__init__()

        self.l2_reg = l2_reg

    def build(self, input_shape):

        self.batch_size = input_shape[0]
        self.vector_size = input_shape[1]
        self.num_channels_input = input_shape[2]

        self.W_CLS = self.add_weight(shape=(1, 1, self.num_channels_input),
                                     initializer="glorot_uniform",
                                     trainable=True,
                                     regularizer=l2(self.l2_reg),
                                     name="W_CLS")

    def call(self, y):

        RES = tf.concat(
            [tf.tile(self.W_CLS, [tf.shape(y)[0], 1, 1]), y], axis=1)

        return RES


def gen_seq_classify(ptb_weight, patch_size, nb_patches, vector_size, num_channels_input_reduced, return_sequences, steps_limit=0, stretch_factor=1, build_backbone=False, consecutive=False, nb_sequences=-1, nb_stacks=1, last_only=False):

    if steps_limit == 0:
        steps_limit = vector_size

    OUTA = []
    print("**********************************************************************************")
    print("last_only:", last_only)
    print("stretch_factor", stretch_factor)
    print("nb_sequences", nb_sequences)
    print("return_sequences", return_sequences)

    nb_occurences = int(vector_size/stretch_factor)
    print("nb_occurences", nb_occurences)
    spots = [((qq+1)*stretch_factor-1) for qq in range(nb_occurences)]
    print("steps_limit", steps_limit)
    print("patch_size", patch_size)
    print("**********************************************************************************")

    if not last_only:

        for step in range(vector_size):  # 35

            if step % 10 == 0:
                print("generating...", step)

            else:

                1

            COLLECTA = []

            for itero in range(nb_patches):

                PTB_STYLE = False

                if np.random.uniform(low=0.0, high=1.0, size=None) < ptb_weight:

                    PTB_STYLE = True

                if PTB_STYLE:

                    randy_B = [step]

                    dispersion = 1.5

                    scale_F = 200

                    while len(randy_B) < patch_size:
                        one_draw = int(np.random.normal(loc=step, scale=int(
                            nb_occurences/scale_F), size=None))
                        if one_draw >= 0 and one_draw < nb_occurences and (one_draw not in randy_B):
                            randy_B.append(one_draw)

                else:

                    randy_B = []

                    dispersion = 1.5

                    scale_F = 200

                    while len(randy_B) < patch_size:
                        gaussian_style = True

                        if gaussian_style:
                            one_draw = int(np.random.normal(loc=step, scale=int(
                                nb_occurences/scale_F), size=None))  
                        else:
                            one_draw = random.randint(0, step) 

                        if one_draw >= 0 and one_draw < nb_occurences and (one_draw not in randy_B):
                            randy_B.append(one_draw)

                first = []

                randy_B = np.asarray(randy_B)
                randy_B = np.sort(randy_B)

                for pp in randy_B:
                    first.append([pp])

                COLLECTA.append(first)

            COLLECTA = np.stack(COLLECTA)

            OUTA.append(COLLECTA)


    for indexuu, elem in enumerate(OUTA):
        print("elem.shape", len(elem))

    OUTA = np.stack(OUTA)

    if OUTA.shape[0] == 1:
        OUTA = np.squeeze(OUTA, axis=0)

    return OUTA
