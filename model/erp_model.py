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
from fast_soft_sort.tf_utils1 import soft_rank, soft_sort
from PIL import Image
from equilib import equi2equi

def create_coordinate(h_out: int, w_out: int) -> np.ndarray:
    """Create mesh coordinate grid with height and width
    return:
    - coordinate (np.ndarray)
    """
    xs = np.linspace(0, w_out - 1, w_out)
    theta = xs * 2 * np.pi / w_out - np.pi
    ys = np.linspace(0, h_out - 1, h_out)
    phi = ys * np.pi / h_out - np.pi / 2
    theta, phi = np.meshgrid(theta, phi)
    coord = np.stack((theta, phi), axis=-1)
    return coord

def create_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    z_down: bool = False,
) -> np.ndarray:
    """Create Rotation Matrix"""
    # calculate rotation about the x-axis
    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )
    # calculate rotation about the y-axis
    if z_down:
        pitch = -pitch
    R_y = np.array(
        [
            [np.cos(pitch), 0.0, -np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    # calculate rotation about the z-axis
    if z_down:
        yaw = -yaw
    R_z = np.array(
        [
            [np.cos(yaw), np.sin(yaw), 0.0],
            [-np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return R_z @ R_y @ R_x

def get_grid(rot, h_out, w_out):
    
    a = create_coordinate(h_out, w_out)  # (theta, phi)s
    norm_A = 1
    x = norm_A * np.cos(a[:, :, 1]) * np.cos(a[:, :, 0])
    y = norm_A * np.cos(a[:, :, 1]) * np.sin(a[:, :, 0])
    z = norm_A * np.sin(a[:, :, 1])
    A = np.stack((x, y, z), axis=-1)

    R = create_rotation_matrix(rot["roll"], rot["pitch"], rot["yaw"], z_down=False)

    A = A[:, :, :, np.newaxis]
    B = R @ A
    B = B.squeeze(3)

    # calculate rotations per perspective coordinates
    phi = np.arcsin(B[:, :, 2] / np.linalg.norm(B, axis=-1))
    theta = np.arctan2(B[:, :, 1], B[:, :, 0])

    # center the image and convert to pixel location
    ui = (theta - np.pi) * w_out / (2 * np.pi)
    uj = (phi - np.pi / 2) * h_out / np.pi
    # out-of-bounds calculations
    ui = np.where(ui < 0, ui + w_out, ui)
    ui = np.where(ui >= w_out, ui - w_out, ui)
    uj = np.where(uj < 0, uj + h_out, uj)
    uj = np.where(uj >= h_out, uj - h_out, uj)
    grid = np.stack((uj, ui), axis=0)
    
    return grid

def _feature_extractor():
    
    input_tensor = Input(shape=(256, 512, 3), name='input_feature')
    # input_tensor = Input(shape=(None, None, 3), name='input_feature')
    x = CHConv(filters = 8, kernel_size = (3, 3), padding = 'same', kernel_regularizer=regularizers.l2(0.01),
               kernel_initializer=HeUniform())(input_tensor)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    x = CHConv(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01),
               kernel_initializer=HeUniform())(input_tensor)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    x = CHConv(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01),
               kernel_initializer=HeUniform())(input_tensor)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    x = CHConv(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01),
               kernel_initializer=HeUniform())(input_tensor)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    x = CHConv(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01),
               kernel_initializer=HeUniform())(input_tensor)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    x = CHConv(filters=1, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01),
               kernel_initializer=HeUniform())(input_tensor)

    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    
    return Model(input_tensor, x)

# @tf.function
@tf.custom_gradient
def keypoints_search(feature_map):
    feature_map = tf.squeeze(feature_map, axis=3)
    feature_map = tf.reshape(feature_map, (-1, 8 * 16))
    rk = soft_sort(feature_map, regularization_strength=1, direction="DESCENDING")
    feature_keypoints = tf.zeros_like(feature_map)
    feature_map = tf.cast(feature_map, tf.float16)
    rk = tf.cast(rk, tf.float16)[0]
    threshold = tf.gather(rk, 64)
    ret = tf.cast(tf.where(tf.less(threshold, feature_map[0])), tf.float32)
    print(ret.shape)
    ret_tile = tf.tile(ret[0:1], [64, 1])
    alt = tf.cast(tf.reshape(tf.keras.backend.arange(64), (64, 1)), tf.float32)
    ret = tf.concat([ret, ret_tile, alt], axis = 0)
    for j in range(64):
        feature_keypoints = tf.tensor_scatter_nd_update(feature_keypoints, [[0, j]], ret[j])
    # feature_keypoints = rk
    
    def grad(v):
        # gradients = tf.cast(v, tf.float32) * feature_map
        gradients = tf.cast(v, tf.float32)
        # print(gradients)
        gradients = tf.reshape(gradients, (-1, 16, 32, 1))
        return gradients
    
    return feature_keypoints, grad

def spatial_softArgmax(features, window = 32, temperature = 0.1):
    
    shape = tf.shape(features)
    height, width, num_channels = shape[1], shape[2], shape[3]
    tmp = []
    for i in range(256 // window):
        for j in range(512 // window):
            
            posx, posy = tf.meshgrid(tf.linspace(-1., 1., num = window), 
                                                 tf.linspace(-1., 1., num = window), indexing='ij')
            posx = tf.reshape(posx, [window * window])
            posy = tf.reshape(posy, [window * window])
            
            features_local = features[:, window * i: window * (i + 1), window * j: window * (j + 1), :]
            features_local = tf.reshape(tf.transpose(features_local, [0, 3, 1, 2]), [-1, window * window])
    
            softmax_attention = tf.nn.softmax(features_local / temperature)
    
            expected_x = tf.reduce_sum(posx * softmax_attention, 1, keepdims = True)
            expected_y = tf.reduce_sum(posy * softmax_attention, 1, keepdims = True)
            
            expected_x = expected_x * 15.5 + 15.5     # [-1,1] -> [0,31]
            expected_y = expected_y * 15.5 + 15.5
            # expected_x = expected_x * 7.5 + 7.5
            # expected_y = expected_y * 7.5 + 7.5
            
            expected_x += i * window
            expected_y +=  j * window
            expected_xy = tf.concat([expected_x, expected_y], axis = 1)
            feature_keypoints_local = tf.reshape(expected_xy, [-1, num_channels * 2])
            
            tmp.append(feature_keypoints_local)
    feature_keypoints = tf.concat(tmp, 1)
    # feature_keypoints = tf.reshape(feature_keypoints, [shape[0], -1, num_channels * 2])
    feature_keypoints = tf.reshape(feature_keypoints, [shape[0], (256 // window) * (512 // window), 2])
    # feature_keypoints = tf.reshape(feature_keypoints, [256, -1, num_channels * 2])
    # return tf.expand_dims(feature_keypoints, 0)
    return feature_keypoints

def pers_crop(input_img_keypoint, out_size=32):
    image = input_img_keypoint[0]
    center_points = input_img_keypoint[1]
    frame_height = image.shape[1]
    frame_width = image.shape[2]
    frame_channel = image.shape[3]
    FOV=[np.pi/24, np.pi/12]
    PI = np.pi
    PI_2 = PI * 0.5
    PI2 = PI * 2
    height = out_size
    width = out_size

    xx, yy = tf.meshgrid(tf.linspace(0, 1, width), tf.linspace(0, 1, height))
    x_t_flat = tf.reshape(xx, (-1, 1))
    y_t_flat = tf.reshape(yy, (-1, 1))
    nfovs = []
    for i in range(64):
        center_point = tf.cast(center_points[i], tf.float64) / tf.cast(tf.convert_to_tensor([256., 512.]), tf.float64)
        screen_points = tf.concat(axis=1, values=[x_t_flat, y_t_flat])
        cp = (center_point * 2 - 1) * np.array([PI_2, PI])
        convertedScreenCoord = (screen_points * 2 - 1) * np.array([PI, PI_2]) * (np.ones(screen_points.shape) * FOV)
        convertedScreenCoord = tf.transpose(convertedScreenCoord)
        x = convertedScreenCoord[0]
        y = convertedScreenCoord[1]
    
        rou = tf.sqrt(x ** 2 + y ** 2)
        c = tf.atan(rou)
        sin_c = tf.sin(c)
        cos_c = tf.cos(c)
    
        lat = tf.asin(cos_c * tf.sin(cp[0]) + (y * sin_c * tf.cos(cp[0])) / rou)
        lon = cp[1] + tf.atan2(x * sin_c, rou * tf.cos(cp[0]) * cos_c - y * tf.sin(cp[0]) * sin_c)
    
        lat = (lat / PI_2 + 1.) * 0.5
        lon = (lon / PI + 1.) * 0.5
    
        screen_coord = tf.concat(axis=1, values=[tf.expand_dims(lon, -1), tf.expand_dims(lat, -1)])
        screen_coord = tf.transpose(screen_coord)
        uf = tf.math.floormod(screen_coord[0],1) * frame_width  # long - width
        vf = tf.math.floormod(screen_coord[1],1) * frame_height  # lat - height
    
        x0 = tf.cast(tf.floor(uf), tf.int64)  # coord of pixel to bottom left
        y0 = tf.cast(tf.floor(vf),tf.int64)
        x2 = tf.add(x0, tf.cast(tf.ones(uf.shape), tf.int64))  # coords of pixel to top right
        y2 = tf.add(y0, tf.cast(tf.ones(vf.shape), tf.int64))
    
        base_y0 = tf.multiply(y0, frame_width)
        base_y2 = tf.multiply(y2, frame_width)
    
        A_idx = tf.math.floormod(tf.add(base_y0, x0), 131072)
        B_idx = tf.math.floormod(tf.add(base_y2, x0), 131072)
        C_idx = tf.math.floormod(tf.add(base_y0, x2), 131072)
        D_idx = tf.math.floormod(tf.add(base_y2, x2), 131072)
    
        flat_img = tf.reshape(image, [-1, frame_channel])

        A = tf.gather(flat_img, A_idx, axis=0)
        B = tf.gather(flat_img, B_idx, axis=0)
        C = tf.gather(flat_img, C_idx, axis=0)
        D = tf.gather(flat_img, D_idx, axis=0)
    
        wa = tf.multiply(tf.cast(x2, tf.float64) - tf.cast(uf, tf.float64), tf.cast(y2, tf.float64) - tf.cast(vf, tf.float64))
        wb = tf.multiply(tf.cast(x2, tf.float64) - tf.cast(uf, tf.float64), tf.cast(vf, tf.float64) - tf.cast(y0, tf.float64))
        wc = tf.multiply(tf.cast(uf, tf.float64) - tf.cast(x0, tf.float64), tf.cast(y2, tf.float64) - tf.cast(vf, tf.float64))
        wd = tf.multiply(tf.cast(uf, tf.float64) - tf.cast(x0, tf.float64), tf.cast(vf, tf.float64) - tf.cast(y0, tf.float64))

        # interpolate
        AA = tf.multiply(tf.cast(A, tf.float64), tf.concat(axis=1, values=[tf.expand_dims(wa, -1), tf.expand_dims(wa, -1), tf.expand_dims(wa, -1)]))
        BB = tf.multiply(tf.cast(B, tf.float64), tf.concat(axis=1, values=[tf.expand_dims(wb, -1), tf.expand_dims(wb, -1), tf.expand_dims(wb, -1)]))
        CC = tf.multiply(tf.cast(C, tf.float64), tf.concat(axis=1, values=[tf.expand_dims(wc, -1), tf.expand_dims(wc, -1), tf.expand_dims(wc, -1)]))
        DD = tf.multiply(tf.cast(D, tf.float64), tf.concat(axis=1, values=[tf.expand_dims(wd, -1), tf.expand_dims(wd, -1), tf.expand_dims(wd, -1)]))

        nfov = tf.reshape(tf.cast(AA + BB + CC + DD, tf.int32), [height, width, 3])
        nfovs.append(tf.expand_dims(nfov, 0))

    return tf.cast(tf.concat(axis=0, values=nfovs), tf.float32)
    
def _erp_detector():
    
    feature_extractor = _feature_extractor()
    
    input_img = Input(shape = (256, 512, 3), name = 'input')
    features = feature_extractor(input_img)
    local_kps = Lambda(spatial_softArgmax, name='spatial_softmax')(features)
    features_down = AvgPool2D((32, 32))(features)
    keypoints_index = Lambda(keypoints_search, name = 'keypoints_index')(features_down)
    keypoints_index = tf.squeeze(tf.cast(tf.gather(keypoints_index, tf.keras.backend.arange(64), axis = 1), tf.int32))
    
    keypoints = tf.gather(local_kps, keypoints_index, axis = 1)
    # keypoints = local_kps
    return Model(inputs=input_img, outputs=features), Model(inputs = input_img, outputs = keypoints, name = 'detector')

def _erp_descriptor():
    
    input_img = Input((256, 512, 3))
    _, detector = _erp_detector()
    kpts = detector(input_img)
    patches = Lambda(pers_crop, arguments={'out_size': 32})([input_img, tf.squeeze(kpts)])
    
    description =  GroupConv2D(filters = 32, kernel_size = 3, group='D4', padding='same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform())(patches)
    description = InstanceNormalization()(description)
    description = Activation('relu')(description)
    
    description =  GroupConv2D(filters = 64, kernel_size = 3, padding = 'same', group="D4", kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform())(description)
    description = InstanceNormalization()(description)
    description = Activation('relu')(description)
    
    description =  GroupConv2D(filters = 128, kernel_size = 3, padding = 'same', group="D4", kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform())(description)
    description = InstanceNormalization()(description)
    description = Activation('relu')(description)
    
    description = MaxPool3D(pool_size=(1,1,description.shape[-2]))(description)
    description = Lambda(lambda x: K.squeeze(x, axis=-2))(description)
    
    description = Flatten()(description)
    description = Dense(units = 256, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform())(description)
    description = InstanceNormalization()(description)
    description = Activation('relu')(description)
    
    description = Dense(units = 128, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform())(description)
    description = InstanceNormalization()(description)
    # reshape_description = Lambda(tf.reshape, arguments = {'shape': [-1, 65, 64]})(description)
    reshape_description = Lambda(tf.reshape, arguments = {'shape': [-1, 64, 128]})(description)
    return detector, Model(inputs=input_img, outputs=reshape_description)