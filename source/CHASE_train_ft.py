from DR_VesselNet import *

import os
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd

from util import *


from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split


tf.config.run_functions_eagerly(True)

desired_size=1008


def scheduler01(epoch, lr):
    if epoch == 50:
        lr = 1e-4

    elif epoch == 100:
        lr = 1e-5

    elif epoch == 120:
        lr = 1e-6

    return lr


def scheduler02(epoch, lr):
    if epoch == 50:
        lr = 1e-4

    return lr


for train_no in range(1, 6):

    x_train = np.load('../Chase/x_train_{:d}.npy'.format(train_no))
    y_train = np.load('../Chase/y_train_{:d}.npy'.format(train_no))
    x_validate = np.load('../Chase/x_validate_{:d}.npy'.format(train_no))
    y_validate = np.load('../Chase/y_validate_{:d}.npy'.format(train_no))

    backbone = DR_VesselNet(input_size=(desired_size, desired_size, 3))
    backbone.compile(optimizer=Adam(lr=1e-3), loss=dice_cross_loss, metrics=['accuracy'])
    weight = "Model_chase/CS_VesselNet0{:d}.h5".format(train_no)

    if os.path.isfile(weight):
        backbone.load_weights(weight)

    model = DR_VesselNet_ft(backbone)
    model.compile(optimizer=Adam(lr=1e-3), loss=dice_cross_loss, metrics=['accuracy'])

    weight_finetune = "Model_chase/CS_VesselNet0{:d}_ft.h5".format(train_no)

    lschedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler02)
    model_checkpoint = ModelCheckpoint(weight_finetune, monitor='val_accuracy', verbose=1, save_best_only=False)

    history = model.fit(x_train, y_train,
                        epochs = 100,
                        batch_size = 1,
                        validation_data = (x_validate, y_validate),
                        shuffle = True,
                        callbacks = [model_checkpoint, lschedule_callback])

    tf.keras.backend.clear_session()

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('Model_csv_chase/CS_VesselNet0{:d}_ft.csv'.format(train_no))

