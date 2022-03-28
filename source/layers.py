import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *


def residual_dense_block(inputs, num_filters):
    eps = 1.1e-5
    x = BatchNormalization( epsilon=eps )( inputs )
    x = Activation('relu')(x)

    x = Conv2D(num_filters, kernel_size=(3, 3),
                               use_bias=False ,
                               padding='same',
                               kernel_initializer='he_normal' )(x)
    x = SpatialDropout2D(0.1)(x)
    out1 = Concatenate()([x, inputs])

    out2 = BatchNormalization( epsilon=eps )(out1)
    out2 = Activation('relu')(out2)

    out2 = Conv2D(num_filters, kernel_size=(3, 3),
                               use_bias=False ,
                               padding='same',
                               kernel_initializer='he_normal' )(out2)
    out2 = SpatialDropout2D(0.1)(out2)
    x_inp = Concatenate()([out1, out2])

    shape = x_inp.shape.as_list()
    out_dim = shape[-1]

    x2 = Conv2D(out_dim, (3, 3),
               padding="same",
               kernel_initializer='he_normal')(x_inp)
    x2 = SpatialDropout2D(0.1)(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    x2 = Conv2D(out_dim, (3, 3),
               padding="same",
               kernel_initializer='he_normal')(x2)
    x2 = SpatialDropout2D(0.1)(x2)

    x2 = Add()([x2, x_inp])
    x2 = ReLU()(x2)

    out = BatchNormalization()(x2)

    return out


def squeeze_excitation_block(input_layer, ratio):

    shape = input_layer.shape.as_list()
    out_dim = shape[-1]

    squeeze = GlobalAveragePooling2D()(input_layer)

    excitation = Dense(units=out_dim / ratio, activation='relu')(squeeze)
    excitation = Dense(out_dim, activation='sigmoid')(excitation)
    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

    scale = multiply([input_layer, excitation])

    shortcut = Conv2D(out_dim, kernel_size=1, strides=1,
                      padding='same', kernel_initializer='he_normal')(input_layer)
    shortcut = SpatialDropout2D(0.1)(shortcut)
    shortcut = BatchNormalization()(shortcut)

    out = add([shortcut, scale])
    out = ReLU()(out)

    return out

