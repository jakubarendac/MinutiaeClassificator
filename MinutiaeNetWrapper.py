import sys, os
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from datetime import datetime
from scipy import misc, ndimage
from keras.optimizers import Adam

sys.path.append(os.path.realpath('./MinutiaeNet/CoarseNet'))
sys.path.append(os.path.realpath('./MinutiaeNet/FineNet'))

from CoarseNet_model import CoarseNetmodel, fuse_minu_orientation
from FineNet_model import FineNetmodel
from CoarseNet_utils import get_maximum_img_size_and_names, label2mnt, py_cpu_nms, nms
from MinutiaeNet_utils import FastEnhanceTexture, get_maps_STFT, fuse_nms, draw_minutiae, show_orientation_field, mnt_writer

coarseNetPath = './MinutiaeNet/Models/CoarseNet.h5'
fineNetPath = './MinutiaeNet/Models/FineNet.h5'
dataPath = './testData/'
output_dir = 'output_CoarseNet'

class MinutiaeNetWrapper:
    def __init__(self):
        # Load CoarseNet model
        self.__coarseNet = CoarseNetmodel((None, None, 1), coarseNetPath, mode='deploy')

        # Load FineNet model
        self.__fineNet = FineNetmodel(num_classes=2,
                               pretrained_path=fineNetPath,
                               input_shape=(224,224,3))

        self.__fineNet.compile(loss='categorical_crossentropy',
                        optimizer=Adam(lr=0),
                        metrics=['accuracy'])

    def readImage(self, imagePath):
        self.__originalImage = np.array(cv2.imread(imagePath,0))
        self.__imageSize = np.array(self.__originalImage.shape, dtype=np.int32) // 8 * 8
        self.__image = self.__originalImage[:self.__imageSize[0], :self.__imageSize[1]]

    def showImage(self):
        cv2.imshow('image',self.__image)
        cv2.waitKey(0)

    def predictImage(self):
        # Generate OF
        textureImg = FastEnhanceTexture(self.__image, sigma=2.5, show=False)
        dirMap, freMap = get_maps_STFT(textureImg, patch_size=64, block_size=16, preprocess=True)
        
        self.__image = np.reshape(self.__image, [1, self.__image.shape[0], self.__image.shape[1], 1])

        enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = self.__coarseNet.predict(self.__image)

        # Use for output mask
        round_seg = np.round(np.squeeze(seg_out))
        seg_out = 1 - round_seg
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_out = cv2.dilate(seg_out, kernel)

        # ========== Adaptive threshold ==================
        final_minutiae_score_threashold = 0.45
        early_minutiae_thres = final_minutiae_score_threashold + 0.05

        # In cases of small amount of minutiae given, try adaptive threshold
        while final_minutiae_score_threashold >= 0:
            mnt = label2mnt(np.squeeze(mnt_s_out) * np.round(np.squeeze(seg_out)), mnt_w_out, mnt_h_out, mnt_o_out,
                            thresh=early_minutiae_thres)

            mnt_nms_1 = py_cpu_nms(mnt, 0.5)
            mnt_nms_2 = nms(mnt)
            # Make sure good result is given
            if mnt_nms_1.shape[0] > 4 and mnt_nms_2.shape[0] > 4:
                break
            else:
                final_minutiae_score_threashold = final_minutiae_score_threashold - 0.05
                early_minutiae_thres = early_minutiae_thres - 0.05

        mnt_nms = fuse_nms(mnt_nms_1, mnt_nms_2)

        mnt_nms = mnt_nms[mnt_nms[:, 3] > early_minutiae_thres, :]
        mnt_refined = []

        # ======= Verify using FineNet ============
        patch_minu_radio = 22
        if fineNetPath != None:
            for idx_minu in range(mnt_nms.shape[0]):
                try:
                    # Extract patch from image
                    x_begin = int(mnt_nms[idx_minu, 1]) - patch_minu_radio
                    y_begin = int(mnt_nms[idx_minu, 0]) - patch_minu_radio
                    patch_minu = self.__originalImage[x_begin:x_begin + 2 * patch_minu_radio,
                                 y_begin:y_begin + 2 * patch_minu_radio]

                    patch_minu = cv2.resize(patch_minu, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)

                    ret = np.empty((patch_minu.shape[0], patch_minu.shape[1], 3), dtype=np.uint8)
                    ret[:, :, 0] = patch_minu
                    ret[:, :, 1] = patch_minu
                    ret[:, :, 2] = patch_minu
                    patch_minu = ret
                    patch_minu = np.expand_dims(patch_minu, axis=0)

                    # Can use class as hard decision
                    # 0: minu  1: non-minu
                    # [class_Minutiae] = np.argmax(model_FineNet.predict(patch_minu), axis=1)
                    #
                    # if class_Minutiae == 0:
                    #     mnt_refined.append(mnt_nms[idx_minu,:])

                    # Use soft decision: merge FineNet score with CoarseNet score
                    [isMinutiaeProb] = self.__fineNet.predict(patch_minu)
                    isMinutiaeProb = isMinutiaeProb[0]
                    # print isMinutiaeProb
                    tmp_mnt = mnt_nms[idx_minu, :].copy()
                    tmp_mnt[3] = (4*tmp_mnt[3] + isMinutiaeProb) / 5
                    mnt_refined.append(tmp_mnt)

                except:
                    mnt_refined.append(mnt_nms[idx_minu, :])
    
        mnt_nms_backup = mnt_nms.copy()
        mnt_nms = np.array(mnt_refined)

        if mnt_nms.shape[0] > 0:
            mnt_nms = mnt_nms[mnt_nms[:, 3] > final_minutiae_score_threashold, :]
        
            final_mask = ndimage.zoom(np.round(np.squeeze(seg_out)), [8, 8], order=0)

