import os
import sys

import numpy as np

from MinutiaeClassificator.ClassifyNetWrapper import ClassifyNetWrapper
from MinutiaeClassificator.utils.image_utils import read_image, show_image
from MinutiaeClassificator.MinutiaeNetWrapper import MinutiaeNetWrapper
from MinutiaeClassificator.MinutiaeNet.CoarseNet.MinutiaeNet_utils import draw_minutiae


class MinutiaeExtractorWrapper:
    def __init__(self, coarse_net_path, fine_net_path, classify_net_path):

        self.__extraction_module = MinutiaeNetWrapper(coarse_net_path, fine_net_path)
        self.__classification_module = ClassifyNetWrapper(classify_net_path)

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






        

