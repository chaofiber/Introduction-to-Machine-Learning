# import pandas as pd
import numpy as np
# import tensorflow.compat.v1 as tf
import tensorflow as tf
from PIL import Image, ImageOps
import argparse
# from Model import Model
# from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, RidgeCV, SGDClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score, f1_score
# from sklearn import preprocessing
import math
import csv
# from xgboost import XGBRegressor
import random
import os
# import tf.keras.applications.InceptionResNetV2
 # import InceptionResNetV2, preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
# train_path = './train_triplets.txt';
# test_path = './test_triplets.txt';
# data_saved = True
# train_list = []
# test_list = []
# with open(train_path) as f:
# 	for line in f:
# 		line = line.rstrip('\n')
# 		inner_list = [elt for elt in line.split(' ')]
# 		train_list.append(inner_list)

# with open(test_path) as f:
#     for line in f:
#         line = line.rstrip('\n')
#         inner_list = [elt for elt in line.split(' ')]
#         test_list.append(inner_list)

# buffer = np.load('buffer_224.npy')

# model = VGG16(weights='imagenet', include_top=False)
# model = VGG16()

model = InceptionResNetV2(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        pooling='avg',
                        input_shape=(299,299,3))

# model.layers.pop()
# model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
all_feature = []
n = 100
for i in range(n):
	print("preprocessing sample: ", i)
	filename = "food/"+ str(i).zfill(5) + ".jpg"
	image_0 = Image.open(filename)
	image_90 = image_0.rotate(90)
	image_180 = image_0.rotate(180)
	image_270 = image_0.rotate(270)
	image_flip = ImageOps.flip(image_0)
	image_mirror = ImageOps.mirror(image_0)

	image = image_0.resize((299,299))
	image = np.expand_dims(image, axis=0)
	img_data = preprocess_input(image)
	Incept_feature = model.predict(img_data)
	all_feature.append(Incept_feature)

	image = image_90.resize((299,299))
	image = np.expand_dims(image, axis=0)
	img_data = preprocess_input(image)
	Incept_feature = model.predict(img_data)
	all_feature.append(Incept_feature)

	image = image_180.resize((299,299))		
	image = np.expand_dims(image, axis=0)
	img_data = preprocess_input(image)
	Incept_feature = model.predict(img_data)
	all_feature.append(Incept_feature)

	image = image_270.resize((299,299))
	image = np.expand_dims(image, axis=0)
	img_data = preprocess_input(image)
	Incept_feature = model.predict(img_data)
	all_feature.append(Incept_feature)

	image = image_flip.resize((299,299))
	image = np.expand_dims(image, axis=0)
	img_data = preprocess_input(image)
	Incept_feature = model.predict(img_data)
	all_feature.append(Incept_feature)

	image = image_mirror.resize((299,299))
	image = np.expand_dims(image, axis=0)
	img_data = preprocess_input(image)
	Incept_feature = model.predict(img_data)
	all_feature.append(Incept_feature)


np_feature = np.array(all_feature)

np_feature = np_feature.reshape((n*6,1536))
print(np_feature.shape)
np.save('InceptionResNetV2feature_augmented',np_feature)
# np.save('VGGfeaure',np_feature)