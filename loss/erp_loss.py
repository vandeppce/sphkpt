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

def get_loss(erp_kpts, vp_kpts, erp_feature, num_kp = 64):
    vp_kpts = tf.cast(tf.convert_to_tensor(vp_kpts), tf.float32)
    erp_kpts = tf.squeeze(erp_kpts)
    dis_min = tf.split(tf.zeros((64, 1)), num_or_size_splits = num_kp, axis=0)
    kp_min = tf.split(tf.zeros_like(erp_kpts), num_or_size_splits = num_kp, axis = 0)
    features = tf.squeeze(erp_feature)
    
    for i in range(num_kp):
        kpe_present = erp_kpts[i]
        min_distance = 1000.0
        minidx_present = 0
        for j in range(12 * 64):
            kpv_present = vp_kpts[j]

            distance_present = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(kpe_present - kpv_present))), tf.float32)
            if tf.less(distance_present, min_distance):
                min_distance = distance_present
                minidx_present = j
        dis_min[i] = min_distance
        kp_min[i] = tf.gather(vp_kpts, minidx_present)
    
    kp_loss = 0.

    features = tf.reshape(features, (256 * 512, -1))

    # Don't need to label the positive or negative pairs!
    for i in range(64):
        kp_loss += dis_min[i] ** 2

    total_loss = kp_loss / 64

    return total_loss