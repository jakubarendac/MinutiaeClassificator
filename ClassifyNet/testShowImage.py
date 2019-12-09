import sys, os
import cv2
import numpy as np

image = cv2.imread("./testData/img_files/crd_0208s_01.png",0)
cv2.imshow('image',image)
cv2.waitKey(0)