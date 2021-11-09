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
from erp_model1 import _erp_detector, _erp_descriptor, pers_crop
from equilib import equi2equi
from erp_model import get_grid
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

scenarios = os.listdir('./data/panoContext_data_vp')
epochs = 100
optimizer = tf.optimizers.Adam(learning_rate=1e-3)

for epoch in range(epochs):
    print("Epochs: {}".format(str(epoch + 1)))
    print("----------------------")
    
    for scenario in scenarios:
        print("Scenario: {}".format(scenario))
        print("----------------------")
    
        images = os.listdir('./data/panoContext_data_vp/' + scenario)
        total_img = len(images)
        
        for i, image in enumerate(images):
            # image = "pano_0019e0a0c8ca0913e543c033a843c58f"
            print("{0}/{1} image: {2}".format(str(i + 1), total_img, image))
            print("----------------------")
            views = os.listdir('./data/panoContext_data_vp/' + scenario + '/' + image)
            for v, view in enumerate(views):
                # view = "views1"
                print("{0}/{1} view: {2}".format(str(v + 1), 12, view))
                print("----------------------")
                
                for j in range(3):
                    training_log = open('./log/viewport/training_log.txt', 'a+')
                    # print("Pair: {}/1".format(int(j + 1)))
                    # print("----------------------")
                    
                    id_1 = np.random.randint(1, 4)
                    # id_1 = 1
                    id_2 = np.random.randint(1, 4)
                    while id_2 == id_1:
                        id_2 = np.random.randint(1, 4)
                    # id_2 = 2
                    # print(id_1, id_2)
                    paras = open('./data/panoContext_data_vp/' + scenario + '/' + image + '/' + 
                                 view + '/imagelist.txt', 'r').readlines()
                        
                    try:
                        patch_1 = yuv2rgb('./data/panoContext_data_vp/' + scenario + '/' + image + '/' + 
                                              view + '/' + image + '_patch_' + str(id_1) + '_128x128x8_cf1.yuv', (128, 128), 1, 0)
                        patch_2 = yuv2rgb('./data/panoContext_data_vp/' + scenario + '/' + image + '/' + 
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
                            [keypoints1, keypoints2, keypoints_sphere1, keypoints_sphere2, description_image1, description_image2, features1, features2] = \
                                                                                siamese([np.expand_dims(patch_1.astype(np.float32), 0), np.expand_dims(patch_2.astype(np.float32), 0), 
                                                                                               np.expand_dims(position_img1, 0), np.expand_dims(position_img2, 0)])
                            loss = get_loss(keypoints1, keypoints2, keypoints_sphere1, keypoints_sphere2, description_image1, description_image2, features1, features2)
                            print(loss)
                            training_log.write("The total loss of Epoch {0} -- {1} -- {2} v -- {3} viewport pairs {4} and {5} is {6}\n". \
                                           format(str(epoch + 1), scenario, image, view, str(id_1), str(id_2), str(loss.numpy())))
                            training_log.close()
                        except Exception as e:
                            training_log.write(str(e) + '\n')
                            training_log.close()
                            continue
                        
                    gradient = tape.gradient(loss, siamese.trainable_variables)
                    cropped_gradients = [tf.clip_by_value(grads, -1., 1.) for grads in gradient if grads is not None]
                    optimizer.apply_gradients(zip(gradient, siamese.trainable_variables))
                        
                # print(loss)
            print("The total loss of image {0} is {1}".format(image, str(loss.numpy())))
            if (i + 1) % 10 == 0:
                siamese.save_weights('./log/viewport/weights_epoch_{0}_img_{1}_loss_{2}'.format(str(epoch + 1), str(i + 1), str(loss.numpy())))