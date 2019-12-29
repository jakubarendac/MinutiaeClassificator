import sys, os
import numpy as np

from keras.optimizers import Adam

sys.path.append(os.path.realpath('./MinutiaeNet/FineNet'))
sys.path.append(os.path.realpath('./ClassifyNet'))

from FineNet_model import FineNetmodel
from ClassifyNet_utils import getMinutiaeTypeFromId

classifyNetPath = '/home/jakub/projects/minutiae-extractor/ClassifyNet/Models/ClassifyNet.h5'

class ClassifyNetWrapper:
    def __init__(self):
        self.classifyNet = FineNetmodel(num_classes=2,
                               pretrained_path=classifyNetPath,
                               input_shape=(224,224,3))

        self.classifyNet.compile(loss='categorical_crossentropy',
                        optimizer=Adam(lr=0),
                        metrics=['accuracy'])
                
    def predictImage(self, image):
        [minutiaeClasses] = self.classifyNet.predict(image)

        numpyMinutiaeClasses = np.array(minutiaeClasses)
        maxValueIndex = np.argmax(numpyMinutiaeClasses)

        return float(maxValueIndex)