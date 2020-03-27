import os
import sys

import numpy as np

from MinutiaeClassificator.ClassifyNetWrapper import ClassifyNetWrapper
from MinutiaeClassificator.utils.image_utils import read_image, draw_minutiae
from MinutiaeClassificator.MinutiaeNetWrapper import MinutiaeNetWrapper

class MinutiaeExtractorWrapper:
    def __init__(self):
        pass

    def __extract_minutiae(self, image, original_image):
        extracted_minutiae = self.__extraction_module.extract_minutiae(image, original_image)

        return extracted_minutiae;

    def __classify_minutiae(self, original_image, extracted_minutiae):
        classified_minutiae = self.__classification_module.classify_minutiae(original_image, extracted_minutiae)

        return classified_minutiae
    
    def get_coarse_net_path(self, coarse_net_path):
        self.__coarse_net_path = coarse_net_path

    def get_fine_net_path(self, fine_net_path):
        self.__fine_net_path = fine_net_path

    def get_classify_net_path(self, classify_net_path):
        self.__classify_net_path = classify_net_path

    def load_extraction_module(self):
        self.__extraction_module = MinutiaeNetWrapper(self.__coarse_net_path, self.__fine_net_path)

    def load_classification_module(self):
        self.__classification_module = ClassifyNetWrapper(self.__classify_net_path)

    def get_classified_minutiae(self, image_path):
        image = read_image(image_path)

        extracted_minutiae = self.__extract_minutiae(image['image'], image['original_image'])
        classified_minutiae = self.__classify_minutiae(image['original_image'], extracted_minutiae)

        image_data = draw_minutiae(image['original_image'], classified_minutiae, 15, True)

        return image_data

    def get_extracted_minutiae(self, image_path):
        image = read_image(image_path)

        extracted_minutiae = self.__extract_minutiae(image['image'], image['original_image'])

        image_data = draw_minutiae(image['original_image'], extracted_minutiae, 15, True)

        return image_data






        

