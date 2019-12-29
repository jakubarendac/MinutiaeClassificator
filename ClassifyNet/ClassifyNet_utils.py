import cv2
import sys, os

from ClassifyNet_constants import ENDING_MINUTIAE, BIFURCATION_MINUTIAE, PATCH_MINU_RADIO, RED_COLOR, GREEN_COLOR

def writeMinutiaePatches(mnt_nms, originalImage, output_dir,imageName):
    os.mkdir(output_dir + "/%sminu/"%(imageName))
    for idx_minu in range(mnt_nms.shape[0]):
        # Extract patch from image
        x_begin = int(mnt_nms[idx_minu, 1]) - PATCH_MINU_RADIO
        y_begin = int(mnt_nms[idx_minu, 0]) - PATCH_MINU_RADIO
        patch_minu = originalImage[x_begin:x_begin + 2 * PATCH_MINU_RADIO,
                     y_begin:y_begin + 2 * PATCH_MINU_RADIO]

        cv2.imwrite("%s/%sminu/minu%s.png"%(output_dir, imageName, idx_minu), patch_minu)
    
def getMinutiaeTypeFromId(minutiaeId):
    switcher = {
        0: ENDING_MINUTIAE,
        1: BIFURCATION_MINUTIAE
    }

    return switcher.get(minutiaeId, None)

def setMinutiaePlotColor(minutiaeType):
    switcher = {
        ENDING_MINUTIAE: RED_COLOR,
        BIFURCATION_MINUTIAE: GREEN_COLOR
    }

    return switcher.get(minutiaeType, RED_COLOR)
