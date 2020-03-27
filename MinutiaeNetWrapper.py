import cv2
import numpy as np

from datetime import datetime
from scipy import misc, ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.optimizers import Adam

from MinutiaeClassificator.MinutiaeNet.CoarseNet.MinutiaeNet_utils import FastEnhanceTexture, get_maps_STFT, fuse_nms
from MinutiaeClassificator.MinutiaeNet.CoarseNet.CoarseNet_utils import label2mnt, py_cpu_nms, nms
from MinutiaeClassificator.MinutiaeNet.FineNet.FineNet_model import FineNetmodel
from MinutiaeClassificator.MinutiaeNet.CoarseNet.CoarseNet_model import CoarseNetmodel, fuse_minu_orientation
from MinutiaeClassificator.ClassifyNet.ClassifyNet_constants import PATCH_MINU_RADIO, INPUT_SHAPE


class MinutiaeNetWrapper:
    def __init__(self, coarse_net_path, fine_net_path):
        # Load CoarseNet model
        self.__coarse_net = CoarseNetmodel(
            (None, None, 1), coarse_net_path, mode='deploy')

        # Load FineNet model
        self.__fine_net = FineNetmodel(num_classes=2,
                                       pretrained_path=fine_net_path,
                                       input_shape=INPUT_SHAPE)

        self.__fine_net.compile(loss='categorical_crossentropy',
                                optimizer=Adam(lr=0),
                                metrics=['accuracy'])

    def extract_minutiae(self, image, original_image):
        # Generate OF
        texture_img = FastEnhanceTexture(image, sigma=2.5, show=False)
        dir_map, fre_map = get_maps_STFT(
            texture_img, patch_size=64, block_size=16, preprocess=True)

        image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = self.__coarse_net.predict(
            image)

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
            mnt = label2mnt(
                np.squeeze(mnt_s_out) *
                np.round(
                    np.squeeze(seg_out)),
                mnt_w_out,
                mnt_h_out,
                mnt_o_out,
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
        for idx_minu in range(mnt_nms.shape[0]):
            try:
                # Extract patch from image
                x_begin = int(mnt_nms[idx_minu, 1]) - PATCH_MINU_RADIO
                y_begin = int(mnt_nms[idx_minu, 0]) - PATCH_MINU_RADIO
                patch_minu = original_image[x_begin:x_begin + 2 *
                                            PATCH_MINU_RADIO, y_begin:y_begin + 2 * PATCH_MINU_RADIO]

                try:
                    patch_minu = cv2.resize(patch_minu, dsize=(
                        224, 224), interpolation=cv2.INTER_NEAREST)
                except Exception as e:
                    # TODO : add some reasonable code here - programme will fail on next step
                    print(str(e))

                ret = np.empty(
                    (patch_minu.shape[0], patch_minu.shape[1], 3), dtype=np.uint8)
                ret[:, :, 0] = patch_minu
                ret[:, :, 1] = patch_minu
                ret[:, :, 2] = patch_minu
                patch_minu = ret
                patch_minu = np.expand_dims(patch_minu, axis=0)

                # Use soft decision: merge FineNet score with CoarseNet score
                [is_minutiae_prob] = self.__fine_net.predict(patch_minu)
                is_minutiae_prob = is_minutiae_prob[0]

                tmp_mnt = mnt_nms[idx_minu, :].copy()
                tmp_mnt[3] = (4*tmp_mnt[3] + is_minutiae_prob) / 5
                mnt_refined.append(tmp_mnt)

            except BaseException:
                mnt_refined.append(mnt_nms[idx_minu, :])

        mnt_nms_backup = mnt_nms.copy()
        mnt_nms = np.array(mnt_refined)

        if mnt_nms.shape[0] > 0:
            mnt_nms = mnt_nms[mnt_nms[:, 3] >
                             final_minutiae_score_threashold, :]

        fuse_minu_orientation(dir_map, mnt_nms, mode=3)

        return mnt_nms
