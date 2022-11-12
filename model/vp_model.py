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
from fast_soft_sort.tf_utils import soft_rank, soft_sort
from PIL import Image

def UpSampling(x, width, height):
    '''
    Resize image to w * h
    '''
    return tf.image.resize(x, (width, height))
def PyramidPool(pyramids):
    '''
    Tensor concatenation alone axis
    '''
    vector = [pyramid for pyramid in pyramids]
    concate = tf.concat(vector, -1)
    return concate
def _feature_extractor():
    
    input_tensor = Input(shape=(None, None, 3), name='input_feature')
    x = Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform(), use_bias=False)(input_tensor)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform(), use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform(), use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform(), use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform(), use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    
    # x = Conv2D(filters = 1, kernel_size = (1, 1), padding = 'same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform(), use_bias=False)(x)
    # x = InstanceNormalization()(x)
    # x = Activation('relu')(x)
    
    return Model(input_tensor, x)

def SphericalProjection(fov, u_deg, v_deg, out_hw):
    [height, width] = out_hw
    wFOV = fov
    hFOV = float(height) / width * wFOV
    
    w_len = np.tan(np.radians(wFOV / 2.0))
    h_len = np.tan(np.radians(hFOV / 2.0))

    x_map = np.ones([height, width], np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len, width), [height, 1])
    z_map = -np.tile(np.linspace(-h_len, h_len, height), [width, 1]).T
    
    D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
    xyz = np.stack((x_map, y_map, z_map), axis = 2) / np.repeat(D[:, :, np.newaxis], 3, axis = 2)
    
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(u_deg))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-v_deg))

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lon = np.arcsin(xyz[:, 2])
    lat = np.arctan2(xyz[:, 1] , xyz[:, 0])

    lat = lat.reshape([height, width]) / np.pi * 180
    lon = -lon.reshape([height, width]) / np.pi * 180
    
    lat = lat / 180 * 255.5 + 255.5
    lon = lon / 90 * 127.5 + 127.5
    
    return [lon, lat]

def recoor(keypoint_and_matrix):
    projection_matrix = tf.squeeze(keypoint_and_matrix[1])  # 2x128x128
    keypoints = tf.cast(keypoint_and_matrix[0], tf.int32)            # 1x64x2
    
    [lon, lat] = tf.split(projection_matrix, num_or_size_splits = 2, axis = 0)
    lon = tf.squeeze(lon)
    lat = tf.squeeze(lat)
    keypoints = tf.reshape(keypoints, (-1, 2))
    keypoints_sphere = tf.zeros_like(keypoints)
    
    keypoints = tf.split(keypoints, num_or_size_splits = 64, axis = 0)   # 64
    keypoints_sphere = tf.split(keypoints_sphere, num_or_size_splits = 64, axis = 0) # 64
    
    for i, keypoint in enumerate(keypoints):
        
        keypoint = tf.squeeze(keypoint)
        keypointPosition = [lon[keypoint[1], keypoint[0]], lat[keypoint[1], keypoint[0]]]
        keypoints_sphere[i] = keypointPosition
    
    concate = tf.concat(keypoints_sphere, 0)
    concate = tf.reshape(concate, (64, 2))     # 64
    return tf.expand_dims(concate, 0)

def _random_choice(inputs, n_samples):
    """
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    """
    # (1, n_states) since multinomial requires 2D logits.
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)

    ind = tf.random.categorical(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

    return tf.gather(inputs, ind, name="random_choice")

# @tf.function
@tf.custom_gradient
def keypoints_search(feature_map):
    feature_map = tf.squeeze(feature_map, axis=3)
    feature_map = tf.reshape(feature_map, (-1, 16 * 16))
    rk = soft_sort(feature_map, regularization_strength=1, direction="DESCENDING")
    feature_keypoints = tf.zeros_like(feature_map)
    feature_map = tf.cast(feature_map, tf.float16)
    rk = tf.cast(rk, tf.float16)[0]
    threshold = tf.gather(rk, 64)
    ret = tf.cast(tf.where(tf.less(threshold, feature_map[0])), tf.float32)
    # print(ret.shape)
    ret_tile = tf.tile(ret[0:1], [64, 1])
    ret = tf.concat([ret, ret_tile], axis = 0)
    for j in range(64):
        feature_keypoints = tf.tensor_scatter_nd_update(feature_keypoints, [[0, j]], ret[j])
    # feature_keypoints = rk
    
    def grad(v):
        # gradients = tf.cast(v, tf.float32) * feature_map
        gradients = tf.cast(v, tf.float32)
        # print(gradients)
        gradients = tf.reshape(gradients, (-1, 16, 16, 1))
        return gradients

    return feature_keypoints, grad

def keypoints_coor(keypoints_index):
    keypoints_index = tf.gather(keypoints_index, tf.keras.backend.arange(64), axis = 1)
    # print(keypoints_index)
    ks = []
    for i in range(1):
        keypoints_present = tf.zeros((1, 64, 2))
        k_idx = keypoints_index[i]
        for j in range(64):
            tmp = k_idx[j]
            coor_x = tmp // 128
            coor_y = tmp % 128
            keypoints_present = tf.tensor_scatter_nd_update(keypoints_present, [[0, j, 0]], [coor_x])
            keypoints_present = tf.tensor_scatter_nd_update(keypoints_present, [[0, j, 1]], [coor_y])
        ks.append(keypoints_present)
    keypoints = tf.concat(ks, axis=0)
    return keypoints

def spatial_softArgmax(features, window = 8, temperature = 0.1):
    
    shape = tf.shape(features)
    height, width, num_channels = shape[1], shape[2], shape[3]
    tmp = []
    for i in range(128 // window):
        for j in range(128 // window):
            
            posx, posy = tf.meshgrid(tf.linspace(-1., 1., num = window), 
                                                 tf.linspace(-1., 1., num = window), indexing='ij')
            posx = tf.reshape(posx, [window * window])
            posy = tf.reshape(posy, [window * window])
            
            features_local = features[:, window * i: window * (i + 1), window * j: window * (j + 1), :]
            features_local = tf.reshape(tf.transpose(features_local, [0, 3, 1, 2]), [-1, window * window])
    
            softmax_attention = tf.nn.softmax(features_local / temperature)
    
            expected_x = tf.reduce_sum(posx * softmax_attention, 1, keepdims = True)
            expected_y = tf.reduce_sum(posy * softmax_attention, 1, keepdims = True)
            
            expected_x = expected_x * 3.5 + 3.5     # [-1,1] -> [0,15]
            expected_y = expected_y * 3.5 + 3.5
            
            expected_x += i * window
            expected_y +=  j * window
            expected_xy = tf.concat([expected_x, expected_y], axis = 1)
            feature_keypoints_local = tf.reshape(expected_xy, [-1, num_channels * 2])
            
            tmp.append(feature_keypoints_local)
    feature_keypoints = tf.concat(tmp, 1)
    feature_keypoints = tf.reshape(feature_keypoints, [shape[0], (128 // window) ** 2, 2])

    return feature_keypoints

def _detector():
    
    feature_extractor = _feature_extractor()
    
    input1 = Input(shape = (128, 128, 3), name = 'input')
    feature1 = feature_extractor(input1)
    
    # input2 = Input(shape = (64, 64, 3), name = 'input2')
    input2 = Lambda(UpSampling, arguments = {'width': 64, 'height': 64})(input1)
    feature2 = Lambda(UpSampling, arguments = {'width': 128, 'height': 128})(feature_extractor(input2))
    
    # input3 = Input(shape = (32, 32, 3), name = 'input3')
    input3 = Lambda(UpSampling, arguments = {'width': 32, 'height': 32})(input2)
    feature3 = Lambda(UpSampling, arguments = {'width': 128, 'height': 128})(feature_extractor(input3))
    
    features = Lambda(PyramidPool, name = 'pyramid')([feature1, feature2, feature3])
    features = Conv2D(filters = 1, kernel_size = (1, 1), padding = 'same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform(), use_bias=False)(features)
    features = InstanceNormalization()(features)
    # features  = BatchNormalization()(features)
    features = Activation('relu')(features)
    
    return Model(inputs=input1, outputs=features, name='detector')

def _keypoint():
    
    input_feature = Input(shape=(128, 128, 1))
    local_kps = Lambda(spatial_softArgmax, name = 'spatial_softmax')(input_feature)
    
    features_down = AvgPool2D(8)(input_feature)
    keypoints_index = Lambda(keypoints_search, name = 'keypoints_index')(features_down)
    keypoints_index = tf.squeeze(tf.cast(tf.gather(keypoints_index, tf.keras.backend.arange(64), axis = 1), tf.int32))
    
    keypoints = tf.gather(local_kps, keypoints_index, axis = 1)
    return Model(inputs=input_feature, outputs = keypoints)
    
def _descriptor():
    
    reshape_patches = Input(shape = (32, 32, 3))
    
    description =  GroupConv2D(filters = 32, kernel_size = 3, group='D4', padding='same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=HeUniform())(reshape_patches)
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
    
    return Model(inputs = reshape_patches, outputs = reshape_description, name = 'descriptor')

def _vp_model():
    
    detector = _detector()
    input_image = Input(shape = (128, 128, 3), name = 'input_img')
    
    input_position = Input(shape = (2, 128, 128), name = 'input_position')  
    
    features = detector(input_image)
    
    keypoint = _keypoint()
    keypoints = keypoint(features)
    reshape_keypoints = Lambda(tf.reshape, arguments = {'shape': [-1, 2]}, name = "reshape_keypoints")(keypoints)
    
    patches  =  Lambda(transformer_crop, arguments  =  {'out_size': 32}, name = 'crop_patches')([input_image, reshape_keypoints])
    reshape_patches = Lambda(tf.reshape, arguments = {'shape': [-1, 32, 32, 3]}, name = 'reshape_patches')(patches)

    
    descriptor = _descriptor()
    
    description_image = descriptor(reshape_patches)
    
    keypoints_sphere = Lambda(recoor, name = 'recoor_kp')([keypoints, input_position])   # (64, 2)
    
    end_to_end_descriptor = Model(inputs = [input_image, input_position], outputs = [keypoints_sphere, description_image])

    return detector, Model(inputs = input_image, outputs = keypoints), descriptor, end_to_end_descriptor