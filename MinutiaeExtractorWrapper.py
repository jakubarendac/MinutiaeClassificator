import os
import sys

import numpy as np

sys.path.append(os.path.realpath('MinutiaeClassificator/utils'))
sys.path.append(os.path.realpath('MinutiaeClassificator/MinutiaeNet/CoarseNet'))
sys.path.append(os.path.realpath('MinutiaeClassificator/MinutiaeNet/FineNet'))

from ClassifyNetWrapper import ClassifyNetWrapper
from image_utils import read_image, show_image
from MinutiaeNetWrapper import MinutiaeNetWrapper
from MinutiaeNet_utils import draw_minutiae


class MinutiaeExtractorWrapper:
    def __init__(self):

        self.__extraction_module = MinutiaeNetWrapper()
        self.__classification_module = ClassifyNetWrapper()

    def __extract_minutiae(self, image, original_image):
        extracted_minutiae = self.__extraction_module.extract_minutiae(image, original_image)

        return extracted_minutiae;

    def __classify_minutiae(self, original_image, extracted_minutiae):
        classified_minutiae = self.__classification_module.classify_minutiae(original_image, extracted_minutiae)

        return classified_minutiae

    def get_classified_minutiae(self, image_path):
        image = read_image(image_path)

        extracted_minutiae = self.__extract_minutiae(image['image'], image['original_image'])
        classified_minutiae = self.__classify_minutiae(image['original_image'], extracted_minutiae)

        return classified_minutiae

    def get_extracted_minutiae(self, image_path):
        image = read_image(image_path)

        extracted_minutiae = self.__extract_minutiae(image['image'], image['original_image'])

        return extracted_minutiae






        

