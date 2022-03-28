import os
import cv2
import numpy as np
import math


from sklearn.metrics import  f1_score, recall_score
from sklearn.metrics import  roc_auc_score, accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import  ModelCheckpoint
from util import *
from sklearn.metrics import matthews_corrcoef

from DR_VesselNet import *

testing_images_loc = '../Drive/test/images/'
testing_label_loc = '../Drive/test/label/'

test_files = os.listdir(testing_images_loc)
test_data = []
test_label = []

desired_size=592

for i in test_files:
    print(i)

    im = cv2.imread(testing_images_loc + i)
    label = cv2.imread(testing_label_loc + i.split('_')[0] + '_manual1.png', cv2.IMREAD_GRAYSCALE)
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

    test_data.append(cv2.resize(new_im, (desired_size, desired_size)))
    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    test_label.append(temp)

train_no = 2

test_data = np.array(test_data)
test_label = np.array(test_label)

x_test = test_data.astype('float32') / 255.
y_test = test_label.astype('float32') / 255.

x_test = np.reshape(x_test, (len(x_test), desired_size, desired_size, 3))
y_test = np.reshape(y_test, (len(y_test), desired_size, desired_size, 1))
y_test=crop_to_shape(y_test,(len(y_test), 584, 565, 1))


sp_all = []
se_all = []
acc_all = []
auc_all = []
gs_all = []

for train_no in range(1, 6):
    backbone = DR_VesselNet(input_size=(desired_size, desired_size, 3))
    backbone.compile(optimizer=Adam(lr=1e-3), loss=dice_cross_loss, metrics=['accuracy'])
    weight = "Model/DR_VesselNet0{:d}.h5".format(train_no)

    if os.path.isfile(weight):
        backbone.load_weights(weight)

    model = DR_VesselNet_ft(backbone)
    model.compile(optimizer=Adam(lr=1e-3), loss=dice_cross_loss, metrics=['accuracy'])

    weight_finetune = "Model/DR_VesselNet0{:d}_ft.h5".format(train_no)
    if os.path.isfile(weight_finetune):
        model.load_weights(weight_finetune)

    start = timer()
    y_pred = model.predict(x_test)
    end = timer()
    print(end - start)
    y_pred = crop_to_shape(y_pred,(20,584,565,1))
    y_pred_threshold = []
    i = 0

    for y in y_pred:
        _, temp = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
        y_pred_threshold.append(temp)

        tempx = temp * 255

        cv2.imwrite('../results/test/drive/{:d}/{:d}.png'.format(train_no, i), tempx.astype(np.uint8))
        np.save('../results/test/drive/{:d}/{:d}.npy'.format(train_no, i), y)

        i += 1


    y_test = list(np.ravel(y_test))

    y_pred_threshold = list(np.ravel(y_pred_threshold))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

    print(tn,tp,fn,fp)
    se = tp / (tp + fn)
    sp = tn	/ (tn + fp)
    ge = np.sqrt(se	* sp)
    mcc = matthews_corrcoef(y_test, y_pred_threshold)
    accs = accuracy_score(y_test, y_pred_threshold)
    aucs = roc_auc_score(y_test, list(np.ravel(y_pred)))

    print('Specificity:', sp)
    print('Sensitivity:', se)
    print('Accuracy:', accs)
    print('AUC:', aucs)
    print('F1Score:', f1_score(y_test, y_pred_threshold))
    print('GScore:', ge)
    print('MCC:', mcc)

    sp_all.append(sp)
    se_all.append(se)
    acc_all.append(accs)
    auc_all.append(aucs)
    gs_all.append(ge)

    tf.keras.backend.clear_session()
    

    
print('sp:'    , np.mean(sp_all), np.std(sp_all))
print('se:'    , np.mean(se_all), np.std(se_all))
print('acc:'   , np.mean(acc_all), np.std(acc_all))
print('auc:'   , np.mean(auc_all), np.std(auc_all))
print('gscore:', np.mean(gs_all), np.std(gs_all))
