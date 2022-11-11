import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy
import cv2
import math
import os
import random
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, Softmax, Flatten, BatchNormalization, Activation, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.layers import MaxPool2D, AvgPool2D, MaxPool3D, AvgPool3D
from tensorflow.keras.initializers import HeUniform
from scipy.ndimage import map_coordinates
from utils.yuv2rgb import yuv2rgb
from utils import InstanceNormalization
from utils.AdvancedLayers import GroupConv2D, CHConv
from fast_soft_sort.tf_utils import soft_rank, soft_sort
from PIL import Image
from model.vp_model import _vp_model, SphericalProjection
from model.erp_model import _erp_detector, _erp_descriptor
from loss.joint_loss import get_vp_loss, get_erp_loss
from utils.equilib import equi2equi
from model.erp_model import get_grid

img = np.array(Image.open('../data/test.jpg').resize((512, 256)))
angle = 0.1

rot = {
        "roll": 0,  #
        "pitch": 0,  # vertical
        "yaw": angle * np.pi,  # horizontal
    }
grid = get_grid(rot, 256, 512)

ro_img = equi2equi(src=np.transpose(img, (2, 0, 1)), rot=rot)
ro_img = np.transpose(ro_img, (1, 2, 0))

# plt.subplot(211)
# plt.imshow(img)
# plt.subplot(212)
# plt.imshow(ro_img)
# plt.show()

ro_img = np.expand_dims(ro_img, 0)

e_featuremap, e_detector = _erp_detector()
v_feature, v_detector, descriptor, v_model = _vp_model()

e_kpts = tf.squeeze(e_detector(img.astype(np.float32)))
e_patches = pers_crop([img, e_kpts])
e_description = np.squeeze(descriptor(e_patches))

e_kpts_ro = tf.squeeze(e_detector(ro_img.astype(np.float32)))
e_patches_ro = pers_crop([ro_img, e_kpts_ro])
e_description_ro = np.squeeze(descriptor(e_patches_ro))