import cv2
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from MinutiaeClassificator.ClassifyNet.ClassifyNet_utils import (getMinutiaeTypeFromId,
                                           setMinutiaePlotColor)


def read_image(image_path):
    original_image = np.array(cv2.imread(image_path, 0))
    image_size = np.array(original_image.shape, dtype=np.int32) // 8 * 8
    image = original_image[:image_size[0], :image_size[1]]

    output = dict()

    output['original_image'] = original_image
    output['image_size'] = image_size
    output['image'] = image

    return output


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)

def draw_minutiae(image, minutiae, r = 15, drawScore = False):
    image = np.squeeze(image)
    fig = plt.figure()
    
    plt.imshow(image,cmap='gray')
    plt.hold(True)
    # Check if no minutiae
    if minutiae.shape[0] > 0:
        for minutiaeData in minutiae:
            x = minutiaeData[0]
            y = minutiaeData[1]
            o = minutiaeData[2]
            s = minutiaeData[3]
            minutiae_type = getMinutiaeTypeFromId(minutiaeData[4])
            minutiae_color = setMinutiaePlotColor(minutiae_type)
            
            plt.plot(x, y, minutiae_color+'s', fillstyle='none', linewidth=1)
            plt.plot([x, x+r*np.cos(o)], [y, y+r*np.sin(o)], minutiae_color+'-')
            if drawScore == True:
                plt.text(x - 10, y - 10, '%.2f' % s, color='yellow', fontsize=4)

    plt.axis([0,image.shape[1],image.shape[0],0])
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=500, bbox_inches='tight', pad_inches = 0)
    buf.seek(0)
    image = Image.open(buf)

    return image

def resize_minutiae_patch(minutiae_patch):
    try:
        minutiae_patch = cv2.resize(minutiae_patch, dsize=(
                        224, 224), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        # TODO : add some reasonable code here - programme will fail on next step
        print(str(e))
                    
    ret = np.empty((minutiae_patch.shape[0], minutiae_patch.shape[1], 3), dtype=np.uint8)
    ret[:, :, 0] = minutiae_patch
    ret[:, :, 1] = minutiae_patch
    ret[:, :, 2] = minutiae_patch
    minutiae_patch = ret
    minutiae_patch = np.expand_dims(minutiae_patch, axis=0)

    return minutiae_patch
