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
from loss.erp_loss import get_loss

scenarios = os.listdir('../data/panoContext_data_erp/')
epochs = 100
optimizer = tf.optimizers.Adam(learning_rate=1e-3)

e_featuremap, e_detector = _erp_detector()
v_feature, v_detector, descriptor, v_model = _vp_model()

for epoch in range(epochs):
    print("Epochs: {}".format(str(epoch + 1)))
    print("----------------------")
    
    for scenario in scenarios:
        print("Scenario: {}".format(scenario))
        print("----------------------")
    
        images = os.listdir('../data/panoContext_data_erp/' + scenario)[:55]
        total_img = len(images)
        
        for i, image in enumerate(images):
            if image[0:4] != 'pano':
                break
            train_log = open('../log/erp/train_logging.txt', 'a+')
            print("{0}/{1} image: {2}".format(str(i + 1), total_img, image))
            print("----------------------")
            
            # group of vp keypoints and feature map
            vp_basedir = '../data/panoContext_data_vp/' + scenario + '/' + image + '/'
            vp_kpts = []
            for i in range(12):
                paras = open(vp_basedir + 'views{}/imagelist.txt'.format(str(i + 1)), 'r').readlines()
                v_id = 1
                para_u = int(paras[v_id - 1].rstrip('\n').split(' ')[1])
                para_v = int(paras[v_id - 1].rstrip('\n').split(' ')[2])
                v_img = yuv2rgb(vp_basedir + 'views{0}/{1}_patch_1_128x128x8_cf1.yuv'.format(str(i + 1), image), (128, 128), 1, 0)
                lon, lat = SphericalProjection(120, para_u, para_v, [128, 128])
                v_kpts2d = np.squeeze(v_detector(np.expand_dims(v_img, 0))).astype(np.uint8)
                for j in range(64):
                    v_kpt = v_kpts2d[j]
                    keypointPosition = [lon[v_kpt[1], v_kpt[0]], lat[v_kpt[1], v_kpt[0]]]
                    vp_kpts.append(keypointPosition)
            vp_kpts = np.array(vp_kpts)
            print(vp_kpts.shape)
            # erp keypoints and feature map
            erp_img = np.array(Image.open('../data/panoContext_data_erp/' + scenario + '/' + image + '/' + image + '.jpg'))
            erp_img = cv2.resize(erp_img, (512, 256))
            
            with tf.GradientTape() as tape:
                erp_kpts = e_detector(np.expand_dims(erp_img.astype(np.float32), 0))
                erp_fms = e_featuremap(np.expand_dims(erp_img.astype(np.float32), 0)) 
                
                loss = get_loss(erp_kpts, vp_kpts, erp_fms)
                print(loss)
                train_log.write("The total loss of Epoch {0} - image {1} is {2}\n".format(str(epoch + 1), image, str(loss.numpy())))
                train_log.close()
            gradient = tape.gradient(loss, e_detector.trainable_variables)
            cropped_gradients = [tf.clip_by_value(grads, -1., 1.) for grads in gradient if grads is not None]
            optimizer.apply_gradients(zip(gradient, e_detector.trainable_variables))
    e_detector.save_weights('../log/erp/weights_epoch_{0}_loss_{1}'.format(str(epoch + 1), str(loss.numpy())))