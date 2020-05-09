import os
import sys

import numpy as np

from MinutiaeClassificator.ClassifyNetWrapper import ClassifyNetWrapper
from MinutiaeClassificator.MinutiaeNetWrapper import MinutiaeNetWrapper
from MinutiaeClassificator.utils.image_utils import read_image, draw_minutiae
from MinutiaeClassificator.ClassifyNet.ClassifyNet_utils import getMinutiaeTypeFromId
from MinutiaeClassificator.exceptions.MinutiaeClassificatorExceptions import CoarseNetPathMissingException, FineNetPathMissingException, ClassifyNetPathMissingException, MinutiaeNetNotLoadedException, ClassifyNetNotLoadedException


class MinutiaeClassificator:
    def __init__(self):
        self.__coarse_net_path = None
        self.__fine_net_path = None
        self.__classify_net_path = None
        self.__extraction_module = None
        self.__classification_module = None

    def __extract_minutiae(self, image, original_image):
        extracted_minutiae = self.__extraction_module.extract_minutiae(
            image, original_image)

        return extracted_minutiae

    def __classify_minutiae_patch(self, minutiae_patch):
        minutiae_patch_type = self.__classification_module.classify_minutiae_patch(minutiae_patch)

        return minutiae_patch_type

    def __classify_minutiae(self, original_image, extracted_minutiae):
        classified_minutiae = self.__classification_module.classify_minutiae(
            original_image, extracted_minutiae)

        return classified_minutiae

    def get_coarse_net_path(self, coarse_net_path):
        self.__coarse_net_path = coarse_net_path

    def get_fine_net_path(self, fine_net_path):
        self.__fine_net_path = fine_net_path

    def get_classify_net_path(self, classify_net_path):
        self.__classify_net_path = classify_net_path

    def load_extraction_module(self):
        if self.__coarse_net_path is None:
            raise CoarseNetPathMissingException

        if self.__fine_net_path is None:
            raise FineNetPathMissingException

        self.__extraction_module = MinutiaeNetWrapper(
            self.__coarse_net_path, self.__fine_net_path)

    def load_classification_module(self):
        if self.__classify_net_path is None:
            raise ClassifyNetPathMissingException

        self.__classification_module = ClassifyNetWrapper(
            self.__classify_net_path)

    def get_extracted_and_classified_minutiae(self, image_path, as_image=True):
        if self.__extraction_module is None:
            raise MinutiaeNetNotLoadedException

        if self.__classification_module is None:
            raise ClassifyNetNotLoadedException

        image = read_image(image_path)

        extracted_minutiae = self.__extract_minutiae(
            image['image'], image['original_image'])
        classified_minutiae = self.__classify_minutiae(
            image['original_image'], extracted_minutiae)

        if as_image:
            image_data = draw_minutiae(
                image['original_image'], classified_minutiae, 15, True)

            return image_data

        return classified_minutiae

    def get_single_classified_minutiae(self, minutiae_patch_path):
        if self.__classification_module is None:
            raise ClassifyNetNotLoadedException

        minutiae_patch = read_image(minutiae_patch_path)

        minutiae_patch_type_id = self.__classify_minutiae_patch(minutiae_patch['original_image'])
        minutiae_patch_type = getMinutiaeTypeFromId(minutiae_patch_type_id)

        return minutiae_patch_type

    def get_classified_minutiae(
            self,
            image_path,
            extracted_minutiae,
            as_image=True):
        if self.__classification_module is None:
            raise ClassifyNetNotLoadedException

        image = read_image(image_path)

        classified_minutiae = self.__classify_minutiae(
            image['original_image'], extracted_minutiae)

        if as_image:
            image_data = draw_minutiae(
                image['original_image'], classified_minutiae, 15, True)

            return image_data

        return classified_minutiae

    def get_extracted_minutiae(self, image_path, as_image=True):
        if self.__extraction_module is None:
            raise MinutiaeNetNotLoadedException

        image = read_image(image_path)

        extracted_minutiae = self.__extract_minutiae(
            image['image'], image['original_image'])

        if as_image:
            image_data = draw_minutiae(
                image['original_image'], extracted_minutiae, 15, True)

            return image_data

        return extracted_minutiae
