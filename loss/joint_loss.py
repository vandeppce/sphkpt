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

def get_vp_loss(keypoints_sphere1, keypoints_sphere2, description_image1, description_image2, e_kpts, e_description, num_kp = 64, features1=None, features2=None):
    kp1 = tf.squeeze(keypoints_sphere1)
    kp2 = tf.squeeze(keypoints_sphere2)
    kpe = tf.squeeze(e_kpts)
    
    description1 =  tf.squeeze(description_image1)
    description2 = tf.squeeze(description_image2)
    description_e = tf.squeeze(e_description)
    
    kp_min = tf.zeros_like(kp1)  # [64, 2]
    kp_min = tf.split(kp_min, num_or_size_splits = num_kp, axis = 0)
    description_min = tf.zeros_like(description1)
    description_min = tf.split(description_min, num_or_size_splits = num_kp, axis = 0)
    dis_min = tf.split(tf.zeros((64, 1)), num_or_size_splits = num_kp, axis=0)
    kp1 = tf.split(kp1, num_or_size_splits = num_kp, axis = 0)
    kp2 = tf.split(kp2, num_or_size_splits = num_kp, axis = 0)
    description2 = tf.split(description2, num_or_size_splits = num_kp, axis = 0)
    
    kpe = tf.split(kpe, num_or_size_splits = num_kp, axis = 0)
    description_e = tf.split(description_e, num_or_size_splits = num_kp, axis = 0)
    
    # vp to vp
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
    concate = K.concatenate(description_min, 0)
    concate = K.reshape(concate, (num_kp, 128))
    concate = description1 - concate

    vp_positive_loss = 0.
    vp_negative_loss = 0.
    margin = 10.
    vp_positive_cnt = 0.
    vp_negative_cnt = 0.
    vp_loss = 0.
    score_loss = 0.
    for i in range(num_kp):
        distance = tf.sqrt(tf.reduce_sum(tf.square(concate[i])))
        if tf.less(dis_min[i], 10):
            vp_positive_loss += distance ** 2
            vp_positive_cnt += 1.
            score_loss += tf.reduce_mean(tf.gather(features1, tf.cast(kp1[i][0, 0] * 128 + kp1[i][0, 1], tf.int64)) +
                tf.gather(features2, tf.cast(kp_min[i][0, 0] * 128 + kp_min[i][0, 1], tf.int64)))
        else:
            vp_negative_loss += (tf.maximum(margin - distance, 0) ** 2)
            vp_negative_cnt += 1.
    
    vp_loss = 0.5 * (vp_positive_loss + vp_negative_loss) / 128 + \
              0.1 * tf.exp(-1 * score_loss / (vp_positive_cnt + 1))
    
    # vp1 to erp
    for i in range(num_kp):
        kp1_present = kp1[i]
        min_distance = 1000.0
        minidx_present = 0

        for j in range(num_kp):
            kpe_present = kpe[j]

            distance_present = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(kp1_present - kpe_present))), tf.float32)
            if tf.less(distance_present, min_distance):
                min_distance = distance_present
                minidx_present = j
        description_min[i] = tf.gather(description_e, minidx_present)
        dis_min[i] = min_distance
    concate = K.concatenate(description_min, 0)
    concate = K.reshape(concate, (num_kp, 128))
    concate = description1 - concate

    ve1_positive_loss = 0.
    ve1_negative_loss = 0.
    margin = 10.
    ve1_positive_cnt = 0.
    ve1_negative_cnt = 0.
    ve1_loss = 0.
    
    for i in range(num_kp):
        distance = tf.sqrt(tf.reduce_sum(tf.square(concate[i])))
        if tf.less(dis_min[i], 10):
            ve1_positive_loss += distance ** 2
            ve1_positive_cnt += 1.
        else:
            ve1_negative_loss += (tf.maximum(margin - distance, 0) ** 2)
            ve1_negative_cnt += 1.
    ve1_loss = (ve1_positive_loss + ve1_negative_loss) / 128
    
    # vp2 to erp
    for i in range(num_kp):
        kp2_present = kp2[i]
        min_distance = 1000.0
        minidx_present = 0

        for j in range(num_kp):
            kpe_present = kpe[j]

            distance_present = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(kp2_present - kpe_present))), tf.float32)
            if tf.less(distance_present, min_distance):
                min_distance = distance_present
                minidx_present = j
        description_min[i] = tf.gather(description_e, minidx_present)
        dis_min[i] = min_distance
    concate = K.concatenate(description_min, 0)
    concate = K.reshape(concate, (num_kp, 128))
    description2 = K.concatenate(description2, 0)
    concate = description2 - concate

    ve2_positive_loss = 0.
    ve2_negative_loss = 0.
    margin = 10.
    ve2_positive_cnt = 0.
    ve2_negative_cnt = 0.
    ve2_loss = 0.
    for i in range(num_kp):
        distance = tf.sqrt(tf.reduce_sum(tf.square(concate[i])))
        if tf.less(dis_min[i], 10):
            ve2_positive_loss += distance ** 2
            ve2_positive_cnt += 1.
        else:
            ve2_negative_loss += (tf.maximum(margin - distance, 0) ** 2)
            ve2_negative_cnt += 1.
            
    ve2_loss = (ve2_positive_loss + ve2_negative_loss) / 128
    # print(vp_positive_cnt, ve1_positive_cnt, ve2_positive_cnt)
    total_loss = 0.
    total_loss = 0.1 * vp_loss + ve1_loss + ve2_loss
    return total_loss
    
def get_erp_loss(erp_kpts, vp_kpts, erp_feature, num_kp = 64):
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