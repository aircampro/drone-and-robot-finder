#
# utility to convert between chaniner format and openCV
#
import numpy as np
import cv2

# Chainer -> OpenCV
def trans_img_cv2(img):
    buf = np.asanyarray(img, dtype=np.uint8).transpose(1, 2, 0)
    dst = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    return dst

# OpenCV -> Chainer
def trans_img_chainer(img):
    buf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = np.asanyarray(buf, dtype=np.float32).transpose(2, 0, 1)
    return dst