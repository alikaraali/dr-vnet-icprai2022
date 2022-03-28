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

train_no = 5

tf.config.run_functions_eagerly(True)

#Publicly available dataset DRIVE is used. Please download it.
training_images_loc = '../Drive/train/images/'
training_label_loc = '../Drive/train/labels/'

train_files = os.listdir(training_images_loc)
train_data = []
train_label = []


desired_size = 592
for i in train_files:
    im = cv2.imread(training_images_loc + i)
    label = cv2.imread(training_label_loc + i.split('_')[0] + '_manual1.png', cv2.IMREAD_GRAYSCALE)

    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    train_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)


train_data = np.array(train_data)
train_label = np.array(train_label)


x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.

#Implementation of https://arxiv.org/abs/2004.03702 is used
x_rotated, y_rotated, x_flipped, y_flipped = img_augmentation(x_train, y_train)

x_train= np.concatenate([x_train, x_rotated,x_flipped])
y_train = np.concatenate([y_train, y_rotated,y_flipped])

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.10, random_state = 101)

x_train = np.reshape(x_train, (len(x_train), desired_size, desired_size, 3))
y_train = np.reshape(y_train, (len(y_train), desired_size, desired_size, 1))

x_validate = np.reshape(x_validate, (len(x_validate), desired_size, desired_size, 3))
y_validate = np.reshape(y_validate, (len(y_validate), desired_size, desired_size, 1))


def scheduler01(epoch, lr):
    if epoch == 50:
        lr = 1e-4

    elif epoch == 100:
        lr = 1e-5

    elif epoch == 120:
        lr = 1e-6

    return lr


model = DR_VesselNet(input_size=(desired_size,desired_size,3))
model.compile(optimizer=Adam(lr=1e-3), loss=dice_cross_loss, metrics=['accuracy'])

weight = "Model/DR_VesselNet0{:d}.h5".format(train_no)

restore = False
if restore and os.path.isfile(weight):
    model.load_weights(weight)


lschedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler01)
model_checkpoint = ModelCheckpoint(weight, monitor='val_accuracy', verbose=1, save_best_only=False)

history = model.fit(x_train, y_train,
                    epochs = 150,
                    batch_size = 2,
                    validation_data = (x_validate, y_validate),
                    shuffle = True,
                    callbacks = [model_checkpoint, lschedule_callback])

tf.keras.backend.clear_session()

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('Model_csv/DR_VesselNet0{:d}.csv'.format(train_no))


def scheduler02(epoch, lr):
    if epoch == 50:
        lr = 1e-4

    return lr


backbone = DR_VesselNet(input_size=(desired_size, desired_size, 3))
backbone.compile(optimizer=Adam(lr=1e-3), loss=dice_cross_loss, metrics=['accuracy'])
weight = "Model/DR_VesselNet0{:d}.h5".format(train_no)

if os.path.isfile(weight):
    backbone.load_weights(weight)

model = DR_VesselNet_ft(backbone)
model.compile(optimizer=Adam(lr=1e-3), loss=dice_cross_loss, metrics=['accuracy'])

weight_finetune = "Model/DR_VesselNet0{:d}_ft.h5".format(train_no)

lschedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler02)
model_checkpoint = ModelCheckpoint(weight_finetune, monitor='val_accuracy', verbose=1, save_best_only=False)

history = model.fit(x_train, y_train,
                    epochs = 100,
                    batch_size = 2,
                    validation_data = (x_validate, y_validate),
                    shuffle = True,
                    callbacks = [model_checkpoint, lschedule_callback])

tf.keras.backend.clear_session()

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('Model_csv/DR_VesselNet0{:d}_ft.csv'.format(train_no))

