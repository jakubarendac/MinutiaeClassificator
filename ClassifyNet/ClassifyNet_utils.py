import cv2
import sys, os

patch_minu_radio = 22

def writeMinutiaePatches(mnt_nms, originalImage, output_dir,imageName):
    os.mkdir(output_dir + "/%sminu/"%(imageName))
    for idx_minu in range(mnt_nms.shape[0]):
        # Extract patch from image
        x_begin = int(mnt_nms[idx_minu, 1]) - patch_minu_radio
        y_begin = int(mnt_nms[idx_minu, 0]) - patch_minu_radio
        patch_minu = originalImage[x_begin:x_begin + 2 * patch_minu_radio,
                     y_begin:y_begin + 2 * patch_minu_radio]

        status = cv2.imwrite("%s/%sminu/minu%s.png"%(output_dir, imageName, idx_minu), patch_minu)
        print("Image written to file-system : ",status) 
    