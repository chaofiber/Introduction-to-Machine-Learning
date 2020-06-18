import pandas as pd
import numpy as np
# import tensorflow as tf
from PIL import Image
import argparse
# from Model import Model
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, RidgeCV, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression,SelectFromModel
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score, f1_score,accuracy_score
from sklearn import preprocessing
import math
import csv
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import random
import os
from sklearn.decomposition import PCA

# def cross_validation(data, Y_train, test):
#     score = 0
#     score_train = 0
#     kfold = 10
#     kf = KFold(n_splits=kfold)
#     alpha = 10
#     weight = 0
#     m, n = data.shape
#     for train_index, val_index in kf.split(data):
#         x_train, x_val = data[train_index], data[val_index]
#         y_train, y_val = Y_train[train_index], Y_train[val_index]
#         #reg = svc(x_train, y_train)
#         reg = xgb(x_train, y_train)        
#         #reg = logistic(x_train,y_train)
#         # y_val_pred = reg.predict_proba(x_val) # shape: (n_sample, n_class)
#         y_val_pred = reg.predict(x_val)
#         # score += roc_auc_score(y_val, y_val_pred[:,1]) * len(y_val)
#         score += f1_score(y_val, y_val_pred) * len(y_val)
        
#         pred = reg.predict(test);
#     # score_train += (mean_squared_error(reg.predict(x_train), y_train)) * len(y_train)
#     # weight += reg.coef_
#     return (score / len(Y_train)), np.array(pred)

def cross_validation(train_list, val_list, test_list, feature_data):

	# we use 10 fold // can do later, choose (0,500),(500,1000),...
	x_train, y_train = get_data(train_list,feature_data)
	x_train = feature_selection(x_train,y_train,400)
	x_val, y_val = get_data(val_list,feature_data)
	x_val = feature_selection(x_val,y_train,400)
	reg = xgb(x_train,y_train)

	y_train_pred = reg.predict(x_train)
	predictions = [round(value) for value in y_train_pred]
	accuracy = accuracy_score(y_train, predictions)
	print("training accuracy", accuracy)

	y_val_pred = reg.predict(x_val)
	predictions = [round(value) for value in y_val_pred]

	accuracy = accuracy_score(y_val, predictions)
	print("validation accuracy", accuracy)

	return



def get_data(train_list, feature_data):
	x = []
	y = []
	for item in train_list:
		sample = np.concatenate((feature_data[int(item[0])],feature_data[int(item[1])],feature_data[int(item[2])]))
		x.append(sample)
		y.append(item[3])
	return np.array(x), np.array(y)

def xgb(X,y):
	xg_reg = XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.3,
	max_depth = 5, alpha = 10, n_estimators = 100);
	# xg_reg = XGBClassifier()
	xg_reg.fit(X,y);
	return xg_reg;


def feature_selection(data,y,num_feature):
	# select = SelectKBest(f_regression, k=num_feature).fit(data,y)
	pca = PCA(n_components=num_feature)
	new_data = pca.fit_transform(data)
	print("pca done")
	print(new_data.shape)
	# select = SelectFromModel(estimator=Lasso(), threshold=-np.inf, max_features=num_feature).fit(data,y)
	# new_data = select.transform(data);
	# idx = select.get_support()
	#print(idx)
	#new_data = np.delete(new_data,idx,1)
	return new_data


def reverse_training_list(data_list):

	random.shuffle(data_list)
	length = len(data_list)
	reversed_data_list = []
	for i in range(length):
		sample = []
		if i<int(length/2.0):
			sample.append(data_list[i][0])
			sample.append(data_list[i][2])
			sample.append(data_list[i][1])
			label = 0
			sample.append(label)
			reversed_data_list.append(sample)
		else:
			sample.append(data_list[i][0])
			sample.append(data_list[i][1])
			sample.append(data_list[i][2])
			label = 1
			sample.append(label)
			reversed_data_list.append(sample)

	random.shuffle(reversed_data_list)

	return reversed_data_list


def split_train_val(train_list,train_ratio):

	train = []
	val = []

	bound = int(5000 * train_ratio)

	for item in train_list:
		if int(item[0])> bound and (int(item[1])>bound) and (int(item[2])>bound):
			val.append(item)

		if int(item[0])<=bound and (int(item[1])<=bound) and (int(item[2])<=bound):
			train.append(item)

	print(len(train))
	print(len(val))
	print(len(train_list))

	return train,val


def main():
    train_path = './train_triplets.txt';
    test_path = './test_triplets.txt';
    data_saved = False
    train_list = []
    test_list = []
    with open(train_path) as f:
        for line in f:
            line = line.rstrip('\n')
            inner_list = [elt for elt in line.split(' ')]
            train_list.append(inner_list)

    with open(test_path) as f:
        for line in f:
            line = line.rstrip('\n')
            inner_list = [elt for elt in line.split(' ')]
            test_list.append(inner_list)


    data = np.load('resnet_18_feat.npy')

    train_list,val_list= split_train_val(train_list,0.8)

    train_list_with_label = reverse_training_list(train_list)
    val_list_with_label = reverse_training_list(val_list)



    print(data.shape)

    cross_validation(train_list_with_label,val_list_with_label,test_list,data)




main()