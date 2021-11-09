import cv2
import numpy as np
from numpy import *
from imageio import imwrite
import matplotlib.pyplot as plt
screenLevels = 255.0
def yuv2rgb(filename, dims, numfrm, startfrm):
    fp = open(filename, 'rb')
    blk_size = int(prod(dims)  * 3 / 2)
    fp.seek(blk_size * startfrm, 0)
    Y = []
    U = []
    V = []
    # print(dims[0])
    # print(dims[1])
    d00 = dims[0] // 2
    d01 = dims[1] // 2
    # print(d00)
    #  print(d01)
    Yt = zeros((dims[0], dims[1]), uint8, 'C')
    Ut = zeros((d00, d01), uint8, 'C')
    Vt = zeros((d00, d01), uint8, 'C')
    for i in range(numfrm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                #print m,n
                Yt[m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                Ut[m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                Vt[m, n] = ord(fp.read(1))
        Y = Y + [Yt]
        U = U + [Ut]
        V = V + [Vt]
    width = dims[1]
    height = dims[0]
    YY = Y[0]
    UU = cv2.resize(U[0], (width, height))
    VV = cv2.resize(V[0], (width, height))
    yuv_img = np.zeros((width, height, 3))
    yuv_img[...,0] = YY
    yuv_img[...,1] = UU
    yuv_img[...,2] = VV
    rgb_img = cv2.cvtColor(yuv_img.astype(np.uint8), cv2.COLOR_YUV2RGB)
    # imwrite('111.png', rgb_img)
    fp.close()
    return rgb_img
    
# width = 128
# height = 128
# rgb_img = yuv2rgb('../panoContext_data_vp/bedroom/pano_aaacisrhqnnvoq/views1/pano_aaacisrhqnnvoq_patch_19_128x128x8_cf1.yuv', (height, width), 1, 0)