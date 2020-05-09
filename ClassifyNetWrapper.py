import os
import sys
import cv2

import numpy as np
from keras.optimizers import Adam

from MinutiaeClassificator.utils.image_utils import resize_minutiae_patch
from MinutiaeClassificator.ClassifyNet.ClassifyNet_model import ClassifyNetModel
from MinutiaeClassificator.ClassifyNet.ClassifyNet_constants import INPUT_SHAPE, NUM_CLASSES, PATCH_MINU_RADIO


class ClassifyNetWrapper:
    def __init__(self, classify_net_path):
        # Load ClassifyNet model
        self.__classifyNet = ClassifyNetModel(num_classes=NUM_CLASSES,
                                              pretrained_path=classify_net_path,
                                              input_shape=INPUT_SHAPE)

        self.__classifyNet.compile(loss='categorical_crossentropy',
                                   optimizer=Adam(lr=0),
                                   metrics=['accuracy'])

    def classify_minutiae_patch(self, minutiae_patch):
        resized_minutiae_patch = resize_minutiae_patch(minutiae_patch)

        [minutiae_classes] = self.__classifyNet.predict(resized_minutiae_patch)
        numpy_minutiae_classes = np.array(minutiae_classes)
        minutiae_type = float(np.argmax(numpy_minutiae_classes))
        
        return minutiae_type

    def classify_minutiae(self, image, extracted_minutiae):
        classified_minutiae = []

        if extracted_minutiae.size != 0:
            for minutiae in range(extracted_minutiae.shape[0]):
                # Extract patch from image
                x_begin = int(extracted_minutiae[minutiae ,1]) - PATCH_MINU_RADIO
                y_begin = int(extracted_minutiae[minutiae ,0]) - PATCH_MINU_RADIO
                patch_minu = image[x_begin:x_begin + 2 * PATCH_MINU_RADIO,
                                            y_begin:y_begin + 2 * PATCH_MINU_RADIO]

                minutiae_type = self.classify_minutiae_patch(patch_minu)

                tmp_mnt = extracted_minutiae[minutiae, :].copy()
                tmp_mnt[4] = minutiae_type

                classified_minutiae.append(tmp_mnt)
        
        return np.array(classified_minutiae)