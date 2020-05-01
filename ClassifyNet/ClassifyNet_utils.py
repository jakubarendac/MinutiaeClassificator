import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

from ClassifyNet_constants import (BIFURCATION_MINUTIAE, BLUE_COLOR,
                                   CYAN_COLOR, ENDING_MINUTIAE,
                                   FRAGMENT_MINUTIAE, GREEN_COLOR,
                                   OTHER_MINUTIAE, ENCLOSURE_MINUTIAE, MAGENTA_COLOR, PATCH_MINU_RADIO, RED_COLOR, CROSSBAR_MINUTIAE, YELLOW_COLOR, INPUT_SHAPE, MINUTIAE_CLASSES, BATCH_SIZE, NUM_CLASSES)
from ClassifyNet_model import plot_confusion_matrix

def writeMinutiaePatches(mnt_nms, originalImage, output_dir, imageName):
    os.mkdir(output_dir + "/%sminu/" % (imageName))
    for idx_minu in range(mnt_nms.shape[0]):
        # Extract patch from image
        x_begin = int(mnt_nms[idx_minu, 1]) - PATCH_MINU_RADIO
        y_begin = int(mnt_nms[idx_minu, 0]) - PATCH_MINU_RADIO
        patch_minu = originalImage[x_begin:x_begin + 2 * PATCH_MINU_RADIO,
                                   y_begin:y_begin + 2 * PATCH_MINU_RADIO]

        cv2.imwrite("%s/%sminu/minu%s.png" %
                    (output_dir, imageName, idx_minu), patch_minu)


def getMinutiaeTypeFromId(minutiaeId):
    switcher = {
        0: ENDING_MINUTIAE,
        1: BIFURCATION_MINUTIAE,
        2: FRAGMENT_MINUTIAE,
        3: ENCLOSURE_MINUTIAE,
        4: CROSSBAR_MINUTIAE,
        5: OTHER_MINUTIAE
    }

    return switcher.get(minutiaeId, None)


def setMinutiaePlotColor(minutiaeType):
    switcher = {
        ENDING_MINUTIAE: RED_COLOR,
        BIFURCATION_MINUTIAE: GREEN_COLOR,
        FRAGMENT_MINUTIAE: BLUE_COLOR,
        ENCLOSURE_MINUTIAE: CYAN_COLOR,
        CROSSBAR_MINUTIAE: YELLOW_COLOR,
        OTHER_MINUTIAE: MAGENTA_COLOR
    }

    return switcher.get(minutiaeType, MAGENTA_COLOR)

def get_confusion_matrix(model, test_data_path, save_image = True):
    """
    This function prepare data for the confusion matrix, create confusion matrix and save to file.
    """

    test_gen = ImageDataGenerator()
    test_batches = test_gen.flow_from_directory(test_data_path, target_size=(
    INPUT_SHAPE[0], INPUT_SHAPE[1]), classes=MINUTIAE_CLASSES, batch_size=BATCH_SIZE)

    score = model.evaluate_generator(test_batches, len(test_batches))
    print 'Test accuracy:', score[1]

    predictions = model.predict_generator(test_batches, len(test_batches), verbose = 1)
    print predictions
    test_labels = test_batches.classes[test_batches.index_array]
    # print test_labels
    # print predictions
    # print np.argmax(predictions, axis = 1)
    # print np.argmax(predictions)
    # # correct = 
    # [[0 1 0 0 0 0]
    #  [0 0 0 0 0 1]
    #  [1 0 0 0 0 0]
    #  [0 0 1 0 0 0]
    #  [0 0 0 1 0 0]
    #  [0 0 0 0 1 0]]


    cm = confusion_matrix(test_labels, np.argmax(predictions, axis = 1))
    #cm_tf = tf.confusion_matrix(test_labels, predictions, num_classes = NUM_CLASSES)

    #print cm
    #print cm_tf

    plot_confusion_matrix(cm, MINUTIAE_CLASSES, title='Confusion Matrix - classifyNet - 30.4 - not normalized', normalize= False, save_image = True)