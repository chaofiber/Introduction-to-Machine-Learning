# import pandas as pd
import numpy as np
# import tensorflow.compat.v1 as tf
import tensorflow as tf
from PIL import Image
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
from keras.models import Model

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

buffer = np.load('buffer_224.npy')

# model = VGG16(weights='imagenet', include_top=False)
model = VGG16()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
all_feature = []

for i in range(10000):
	print("preprocessing sample: ", i)
	image = buffer[i]
	image = np.expand_dims(image, axis=0)
	img_data = preprocess_input(image)
	vgg16_feature = model.predict(img_data)
	# print(vgg16_feature.shape)
	all_feature.append(vgg16_feature)
np_feature = np.array(all_feature)
np_feature = np_feature.reshape((10000,4096))
print(np_feature.shape)
np.save('VGGfeature',data)