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
from yuv2rgb import yuv2rgb
from transformer_crop import transformer_crop
from utils import InstanceNormalization
from AdvancedLayers import GroupConv2D
from fast_soft_sort.tf_utils import soft_rank, soft_sort
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_loss(keypoints1, keypoints2, keypoints_sphere1, keypoints_sphere2, description_image1, description_image2, features1, features2, num_kp=64):
    k1 = tf.squeeze(keypoints1)
    k2 = tf.squeeze(keypoints2)
    kp1 = tf.squeeze(keypoints_sphere1) # [1,]
    kp2 = tf.squeeze(keypoints_sphere2)
    description1 = tf.squeeze(description_image1)
    description2 = tf.squeeze(description_image2)
    
    kp_min = tf.zeros_like(kp1)  # [64, 2]
    des_sign = []
    kp_min = tf.split(kp_min, num_or_size_splits = num_kp, axis = 0)
    description_min = tf.zeros_like(description1)
    description_min = tf.split(description_min, num_or_size_splits = num_kp, axis = 0)
    dis_min = tf.split(tf.zeros((64, 1)), num_or_size_splits = num_kp, axis=0)
    kp1 = tf.split(kp1, num_or_size_splits = num_kp, axis = 0)
    kp2 = tf.split(kp2, num_or_size_splits = num_kp, axis = 0)
    k1 = tf.split(k1, num_or_size_splits = num_kp, axis = 0)
    k2 = tf.split(k2, num_or_size_splits = num_kp, axis = 0)
    description2 = tf.split(description2, num_or_size_splits = num_kp, axis = 0)
    
    for i in range(num_kp):
        kp1_present = kp1[i]
        min_distance = 1000.0
        minidx_present = 0

        for j in range(num_kp):
            kp2_present = kp2[j]

            distance_present = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(kp1_present - kp2_present))), tf.float32)
            if tf.less(distance_present, min_distance):
                min_distance = distance_present
                minidx_present = j
        description_min[i] = tf.gather(description2, minidx_present)
        dis_min[i] = min_distance
        kp_min[i] = tf.gather(k2, minidx_present)
        
    concate = K.concatenate(description_min, 0)
    concate = K.reshape(concate, (num_kp, 128))
    concate = description1 - concate

    total_loss = 0.
    positive_loss = 0.
    negative_loss = 0.
    margin = 10.
    positive_cnt = 0.
    negative_cnt = 0.
    positive_dis = 0.
    
    features1 = tf.squeeze(features1)
    features2 = tf.squeeze(features2)
    features1 = tf.reshape(features1, (128 * 128, -1))
    features2 = tf.reshape(features2, (128 * 128, -1))
    score_loss = 0.
    
    for i in range(64):
        distance = tf.sqrt(tf.reduce_sum(tf.square(concate[i])))
        if tf.less(dis_min[i], 10):
            positive_loss += distance ** 2
            positive_cnt += 1.
            score_loss += tf.reduce_mean(tf.gather(features1, tf.cast(k1[i][0, 0] * 128 + kp1[i][0, 1], tf.int64)) + tf.gather(features2, tf.cast(kp_min[i][0, 0] * 128 + kp_min[i][0, 1], tf.int64)))
        else:
            negative_loss += (tf.maximum(margin - distance, 0) ** 2)
            negative_cnt += 1.

    kp_loss = 1 / (tf.maximum(positive_cnt - 10, 0) + 0.01)
    # kp_loss = tf.exp(10 - positive_cnt)
    des_loss = (negative_loss + positive_loss) / 128
    
    # total_loss = des_loss + kp_loss + score_loss
    print(score_loss)
    total_loss = des_loss + tf.exp(-1 * score_loss / (positive_cnt + 1))
    return total_loss