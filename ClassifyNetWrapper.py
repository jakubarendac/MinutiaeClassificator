import os
import sys

import numpy as np
from keras.optimizers import Adam

from ClassifyNet_utils import getMinutiaeTypeFromId
from ClassifyNet_model import ClassifyNetModel
from ClassifyNet_constants import NUM_CLASSES, INPUT_SHAPE

sys.path.append(os.path.realpath('./ClassifyNet'))

classifyNetPath = '/home/jakub/projects/minutiae-extractor/ClassifyNet/Models/ClassifyNet.h5'


class ClassifyNetWrapper:
    def __init__(self):
        # Load ClassifyNet model
        self.__classifyNet = ClassifyNetModel(num_classes=NUM_CLASSES,
                                            pretrained_path=classifyNetPath,
                                            input_shape=INPUT_SHAPE)

        self.__classifyNet.compile(loss='categorical_crossentropy',
                                 optimizer=Adam(lr=0),
                                 metrics=['accuracy'])

    def predictImage(self, image):
        [minutiaeClasses] = self.__classifyNet.predict(image)

        numpyMinutiaeClasses = np.array(minutiaeClasses)
        maxValueIndex = np.argmax(numpyMinutiaeClasses)

        return float(maxValueIndex)
