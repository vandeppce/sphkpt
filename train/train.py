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
from utils import InstanceNormalization
from AdvancedLayers import GroupConv2D
from fast_soft_sort.tf_utils import soft_rank, soft_sort
from PIL import Image
from vp_model import _vp_model, SphericalProjection
from erp_model import _erp_detector, _erp_descriptor, pers_crop
from equilib import equi2equi
from erp_model import get_grid
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

e_featuremap, e_detector = _erp_detector()
v_feature, v_detector, descriptor, v_model = _vp_model()

scenarios = os.listdir('../data/panoContext_data_vp')
epochs = 100
optimizer = tf.optimizers.Adam(learning_rate=1e-3)

for epoch in range(epochs):
    print("Epochs: {}".format(str(epoch + 1)))
    print("----------------------")
    
    for scenario in scenarios:
        print("Scenario: {}".format(scenario))
        print("----------------------")
    
        images = os.listdir('../data/panoContext_data_vp/' + scenario)[:50]
        total_img = len(images)
        
        for i, image in enumerate(images):
            print("{0}/{1} image: {2}".format(str(i + 1), total_img, image))
            print("----------------------")
            
            # train vp
            # using erp keypoints and descs to train vp model
            erp_img = np.array(Image.open('../data/panoContext_data_erp/{0}/{1}/{1}.jpg'.format(scenario, image)))
            erp_img = np.expand_dims(cv2.resize(erp_img, (512, 256)), 0)
            e_kpts = tf.squeeze(e_detector(erp_img.astype(np.float32)))
            e_patches = pers_crop([erp_img, e_kpts])
            e_description = descriptor(e_patches)
            
            views = os.listdir('../data/panoContext_data_vp/' + scenario + '/' + image)
            print("Train the viewport model")
            for v, view in enumerate(views):
                # view = "views1"
                print("{0}/{1} view: {2}".format(str(v + 1), 12, view))
                
                for j in range(1):
                    train_log = open('../log/uniform/train_logging.txt', 'a+')
                    
                    id_1 = np.random.randint(1, 10)
                    id_2 = np.random.randint(1, 10)
                    while id_2 == id_1:
                        id_2 = np.random.randint(1, 10)
                    paras = open('../data/panoContext_data_vp/' + scenario + '/' + image + '/' + 
                                 view + '/imagelist.txt', 'r').readlines()
                        
                    try:
                        patch_1 = yuv2rgb('../data/panoContext_data_vp/' + scenario + '/' + image + '/' + 
                                              view + '/' + image + '_patch_' + str(id_1) + '_128x128x8_cf1.yuv', (128, 128), 1, 0)
                        patch_2 = yuv2rgb('../data/panoContext_data_vp/' + scenario + '/' + image + '/' + 
                                              view + '/' + image + '_patch_' + str(id_2) + '_128x128x8_cf1.yuv', (128, 128), 1, 0)
                    except:
                        print("Invalid image!")
                        continue
                            
                    para_1_u = int(paras[id_1 - 1].rstrip('\n').split(' ')[1])
                    para_1_v = int(paras[id_1 - 1].rstrip('\n').split(' ')[2])
                    lon1, lat1 = SphericalProjection(120, para_1_u, para_1_v, [128, 128])
                    position_img1 = np.concatenate([np.expand_dims(lon1, 0), np.expand_dims(lat1, 0)], 0)

                    para_2_u = int(paras[id_2 - 1].rstrip('\n').split(' ')[1])
                    para_2_v = int(paras[id_2 - 1].rstrip('\n').split(' ')[2])
                    lon2, lat2 = SphericalProjection(120, para_2_u, para_2_v, [128, 128])
                    position_img2 = np.concatenate([np.expand_dims(lon2, 0), np.expand_dims(lat2, 0)], 0)
                        
                    with tf.GradientTape() as tape:
                        try:
                            [keypoints_sphere1, description_image1] = v_model([np.expand_dims(patch_1.astype(np.float32), 0), np.expand_dims(position_img1, 0)])
                            [keypoints_sphere2, description_image2] = v_model([np.expand_dims(patch_2.astype(np.float32), 0), np.expand_dims(position_img2, 0)])
                            v_loss = get_vp_loss(keypoints_sphere1, keypoints_sphere2, description_image1, description_image2, e_kpts, e_description)
                            print(v_loss)
                            train_log.write("The total loss of Epoch {0} -- {1} -- {2} v -- {3} viewport pairs {4} and {5} is {6}\n". \
                                           format(str(epoch + 1), scenario, image, view, str(id_1), str(id_2), str(v_loss.numpy())))
                            train_log.close()
                        except Exception as e:
                            train_log.write(str(e) + '\n')
                            train_log.close()
                            continue
                        
                    gradient = tape.gradient(v_loss, v_model.trainable_variables)
                    cropped_gradients = [tf.clip_by_value(grads, -1., 1.) for grads in gradient if grads is not None]
                    optimizer.apply_gradients(zip(gradient, v_model.trainable_variables))
            
            # train erp
            # using vp kps to train erp model
            print("----------------------")
            print("Train the panorama model")
            vp_basedir = '../data/panoContext_data_vp/' + scenario + '/' + image + '/'
            vp_kpts = []
            for i in range(12):
                paras = open(vp_basedir + 'views{}/imagelist.txt'.format(str(i + 1)), 'r').readlines()
                v_id = np.random.randint(1, 10)
                para_u = int(paras[v_id - 1].rstrip('\n').split(' ')[1])
                para_v = int(paras[v_id - 1].rstrip('\n').split(' ')[2])
                v_img = yuv2rgb(vp_basedir + 'views{0}/{1}_patch_{2}_128x128x8_cf1.yuv'.format(str(i + 1), image, str(v_id)), (128, 128), 1, 0)
                lon, lat = SphericalProjection(120, para_u, para_v, [128, 128])
                v_kpts2d = np.squeeze(v_detector(np.expand_dims(v_img, 0))).astype(np.uint8)
                for j in range(64):
                    v_kpt = v_kpts2d[j]
                    keypointPosition = [lon[v_kpt[1], v_kpt[0]], lat[v_kpt[1], v_kpt[0]]]
                    vp_kpts.append(keypointPosition)
            vp_kpts = np.array(vp_kpts)
        
            with tf.GradientTape() as tape:
                erp_kpts = e_detector(erp_img.astype(np.float32))
                erp_fms = e_featuremap(erp_img.astype(np.float32)) 
                
                p_loss = get_erp_loss(erp_kpts, vp_kpts, erp_fms)
                print(p_loss)
                train_log = open('../log/uniform/train_logging.txt', 'a+')
                train_log.write("The total loss of Epoch {0} - image {1} is {2}\n".format(str(epoch + 1), image, str(p_loss.numpy())))
                train_log.close()
            gradient = tape.gradient(p_loss, e_detector.trainable_variables)
            cropped_gradients = [tf.clip_by_value(grads, -1., 1.) for grads in gradient if grads is not None]
            optimizer.apply_gradients(zip(gradient, e_detector.trainable_variables))
    
    v_model.save_weights('../log/uniform/vp_weights_epoch_{0}_loss_{1}'.format(str(epoch + 1), str(v_loss.numpy())))
    e_detector.save_weights('../log/uniform/erp_weights_epoch_{0}_loss_{1}'.format(str(epoch + 1), str(p_loss.numpy())))