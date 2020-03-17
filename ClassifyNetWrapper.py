import os
import sys
import cv2

import numpy as np
from keras.optimizers import Adam

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

    def classify_minutiae(self, image, extracted_minutiae):
        classified_minutiae = []

        if extracted_minutiae.size != 0:
            for minutiae in range(extracted_minutiae.shape[0]):
                # Extract patch from image
                x_begin = int(extracted_minutiae[minutiae ,1]) - PATCH_MINU_RADIO
                y_begin = int(extracted_minutiae[minutiae ,0]) - PATCH_MINU_RADIO
                patch_minu = image[x_begin:x_begin + 2 * PATCH_MINU_RADIO,
                                            y_begin:y_begin + 2 * PATCH_MINU_RADIO]

                patch_minu = cv2.resize(patch_minu, dsize=(
                   224, 224), interpolation=cv2.INTER_NEAREST)
                ret = np.empty((patch_minu.shape[0], patch_minu.shape[1], 3), dtype=np.uint8)
                ret[:, :, 0] = patch_minu
                ret[:, :, 1] = patch_minu
                ret[:, :, 2] = patch_minu
                patch_minu = ret
                patch_minu = np.expand_dims(patch_minu, axis=0)

                [minutiae_classes] = self.__classifyNet.predict(patch_minu)
                numpy_minutiae_classes = np.array(minutiae_classes)
                minutiae_type = float(np.argmax(numpy_minutiae_classes))

                tmp_mnt = extracted_minutiae[minutiae, :].copy()
                tmp_mnt[4] = minutiae_type

                classified_minutiae.append(tmp_mnt)
        
        return np.array(classified_minutiae)