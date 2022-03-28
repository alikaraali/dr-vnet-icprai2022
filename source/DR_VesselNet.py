'''
Implementation of DR-VNet. Please cite accordingly.
'''

from layers import *

import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

import tensorflow.keras.backend as K


def dice_cross_loss(y_true, y_pred):
    loss1 = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    loss2 = 1 - dice_coef(y_true, y_pred)

    return loss1 + 0.5*loss2


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def ssim_loss(y_true, y_pred):
    loss1 = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    loss2 = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

    return loss2 + loss1


def DR_VesselNet(input_size=(240, 240, 3)):
    input_layer = Input(input_size)

    conv1 = residual_dense_block(input_layer, num_filters = 8)
    conv1 = squeeze_excitation_block(conv1, 2)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = residual_dense_block(pool1, num_filters = 8)
    conv2 = squeeze_excitation_block(conv2, 2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = residual_dense_block(pool2, num_filters = 16)
    conv3 = squeeze_excitation_block(conv3, 2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    convm = residual_dense_block(pool3, num_filters=32)
    convm = squeeze_excitation_block(convm, 2)

    deconv3 = Conv2DTranspose(67, (3, 3), strides=(2, 2), padding="same")(convm)  # (, 56)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = residual_dense_block(uconv3, num_filters = 8)
    uconv3 = squeeze_excitation_block(uconv3, 2)

    deconv2 = Conv2DTranspose(35, (3, 3), strides=(2, 2), padding="same")(uconv3)  # (, 56)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = residual_dense_block(uconv2, num_filters = 4)
    uconv2 = squeeze_excitation_block(uconv2, 2)

    deconv1 = Conv2DTranspose(19, (3, 3), strides=(2, 2), padding="same")(uconv2)  # (, 56)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = residual_dense_block(uconv1, num_filters = 2)
    uconv1 = squeeze_excitation_block(uconv1, 2)

    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    model = Model(inputs=input_layer, outputs=output_layer)
    # model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def DR_VesselNet_ft(backbone):
    '''

    :return:
    '''
    for layer in backbone.layers:
        layer.trainable = False

    conv1a = residual_dense_block(backbone.input, num_filters = 2)
    conv1a = squeeze_excitation_block(conv1a, 2)
    
    conv1b = residual_dense_block(backbone.output, num_filters = 2)
    conv1b = squeeze_excitation_block(conv1b, 2)
    
    concat_layer = concatenate([conv1a, conv1b])
    
    conv1 = residual_dense_block(concat_layer, num_filters = 2)
    conv1 = squeeze_excitation_block(conv1, 2)
    conv1 = residual_dense_block(conv1, num_filters = 2)
    conv1 = squeeze_excitation_block(conv1, 2)
    
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(conv1)

    model = Model(inputs=backbone.input, outputs=output_layer)

    return model


if __name__ == "__main__":
    desired_size = 592

    model = denseResUnet01(input_size=(desired_size, desired_size, 3))
    model.summary()
