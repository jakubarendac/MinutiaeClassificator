import os
import sys

import numpy as np

from ClassifyNetWrapper import ClassifyNetWrapper
from utils.image_utils import read_image, draw_minutiae
from MinutiaeNetWrapper import MinutiaeNetWrapper

class MinutiaeExtractorWrapper:
    def __init__(self):
        pass

    def __extract_minutiae(self, image, original_image, should_get_time):
        extracted_minutiae = self.__extraction_module.extract_minutiae(image, original_image, should_get_time)

        return extracted_minutiae;

    def __classify_minutiae(self, original_image, extracted_minutiae, should_get_time):
        classified_minutiae = self.__classification_module.classify_minutiae(original_image, extracted_minutiae, should_get_time)

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

    def get_classified_minutiae(self, image_path, as_image = True, should_get_time = False):
        image = read_image(image_path)
        
        try:
            extracted_minutiae = self.__extract_minutiae(image['image'], image['original_image'], should_get_time)
            classified_minutiae = self.__classify_minutiae(image['original_image'], extracted_minutiae['minutiae'], should_get_time)

            if as_image:
                image_data = draw_minutiae(image['original_image'], classified_minutiae['minutiae'], 15, True)

                return image_data

            if should_get_time:
                extraction_time = extracted_minutiae['time_elapsed']
                classification_time = classified_minutiae['time_elapsed']

                time_elapsed = extraction_time + classification_time

                classified_minutiae['time_elapsed'] = time_elapsed

            return classified_minutiae

        except ValueError as error:
            print error
            
            return None

    def get_extracted_minutiae(self, image_path, as_image = True, should_get_time = False):
        image = read_image(image_path)

        try:
            extracted_minutiae = self.__extract_minutiae(image['image'], image['original_image'], should_get_time)

            if as_image:
                image_data = draw_minutiae(image['original_image'], extracted_minutiae['minutiae'], 15, True)

                return image_data

            return extracted_minutiae

        except ValueError as error:
            print error

            return None






        

