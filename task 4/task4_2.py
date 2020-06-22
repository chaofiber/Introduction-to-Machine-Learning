import pandas as pd
import numpy as np
# import tensorflow as tf
from PIL import Image
import argparse
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score, f1_score,accuracy_score
np.random.seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras import regularizers
import os
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
parser = argparse.ArgumentParser()

parser.add_argument('--epoch',  type=int, default=100, help='num of epochs')
args = parser.parse_args()

def cross_validation(train_list, val_list, test_list, feature_data):

	# we use 10 fold // can do later, choose (0,500),(500,1000),...
	x_train, y_train = get_data(train_list,feature_data)
	x_val, y_val = get_data(val_list,feature_data)
	reg_10 = K_MLP(x_train,y_train,15)
	reg_20 = K_MLP(x_train,y_train,25)
	reg_30 = K_MLP(x_train,y_train,35)

	# eclf = VotingClassifier(estimators=[('10',clf1),('20',clf2),('30',clf3)],
 #                        voting='soft',weights=[1,1,1])
	# eclf.fit(x_train,y_train)
	y_train_pred1 = reg_10.predict_classes(x_train)
	y_train_pred2 = reg_20.predict_classes(x_train)
	y_train_pred3 = reg_30.predict_classes(x_train)
	predictions1 = [np.round(value) for value in y_train_pred1]
	predictions2 = [np.round(value) for value in y_train_pred2]
	predictions3 = [np.round(value) for value in y_train_pred3]
	predictions = []
	for i in range(len(predictions1)):
		if predictions1[i]+predictions2[i]+predictions3[i]>1.5:
			predictions.append(1.0)
		else:
			predictions.append(0.0)
	accuracy = accuracy_score(y_train, predictions)
	print("training accuracy", accuracy)

	y_val_pred1 = reg_10.predict_classes(x_val)
	y_val_pred2 = reg_20.predict_classes(x_val)
	y_val_pred3 = reg_30.predict_classes(x_val)
	predictions1 = [np.round(value) for value in y_val_pred1]
	predictions2 = [np.round(value) for value in y_val_pred2]
	predictions3 = [np.round(value) for value in y_val_pred3]
	predictions = []
	for i in range(len(predictions1)):
		if predictions1[i]+predictions2[i]+predictions3[i]>1.5:
			predictions.append(1.0)
		else:
			predictions.append(0.0)
	# predictions = [np.round(value) for value in y_val_pred]

	accuracy = accuracy_score(y_val, predictions)
	print("validation accuracy", accuracy)

	test_data = get_test_data(test_list,feature_data)
	y_test1 = reg_10.predict_classes(test_data)
	y_test2 = reg_20.predict_classes(test_data)
	y_test3 = reg_30.predict_classes(test_data)
	predictions1 = [np.round(value) for value in y_test1]
	predictions2 = [np.round(value) for value in y_test2]
	predictions3 = [np.round(value) for value in y_test3]
	predictions = []
	for i in range(len(predictions1)):
		if predictions1[i]+predictions2[i]+predictions3[i]>1.5:
			predictions.append(1.0)
		else:
			predictions.append(0.0)
	# y_test = reg.predict(test_data)
	# predictions = [np.round(value) for value in y_test]

	with open("file.txt", "w") as output:
		for row in predictions:
			# output.write(str(int(row[0]))+'\n')
			output.write(str(int(row))+'\n')


	return

def K_MLP(X,y,num_epoch):

	model = Sequential()
	model.add(Dense(2048,input_dim = 1536*3, activation = 'relu'))
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.summary()
	#pt = keras.optimizers.Adam(learning_rate=0.001)
	opt = keras.optimizers.Adadelta()
	#opt = keras.optimizers.Nadam()
	model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
	model.fit(X,y,epochs=num_epoch,batch_size=1024)
	return model

def get_test_data(test_list,feature_data):
	x = []
	for item in test_list:
		sample = np.concatenate((feature_data[int(item[0])],feature_data[int(item[1])],feature_data[int(item[2])]))
		x.append(sample)
	return np.array(x)

def get_data(train_list, feature_data):
	x = []
	y = []
	for item in train_list:
		sample = np.concatenate((feature_data[int(item[0])],feature_data[int(item[1])],feature_data[int(item[2])]))
		x.append(sample)
		y.append(item[3])
	return np.array(x), np.array(y)

def reverse_training_list(data_list):

	np.random.shuffle(data_list)
	length = len(data_list)
	reversed_data_list = []
	for i in range(length):
		sample = []
		sample.append(data_list[i][0])
		sample.append(data_list[i][2])
		sample.append(data_list[i][1])
		label = 0
		sample.append(label)
		reversed_data_list.append(sample)
		sample = []
		sample.append(data_list[i][0])
		sample.append(data_list[i][1])
		sample.append(data_list[i][2])
		label = 1
		sample.append(label)
		reversed_data_list.append(sample)

	np.random.shuffle(reversed_data_list)
	print("double size:", len(reversed_data_list))

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
	train_path = './train_triplets.txt'
	test_path = './test_triplets.txt'
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
	
	
	# data = np.load('resnet_18_feat.npy')
	# data = np.load('VGGfeature.npy')
	data = np.load('InceptionResNetV2feature.npy')
	
	train_list,val_list= split_train_val(train_list,0.95)
	
	train_list_with_label = reverse_training_list(train_list)
	val_list_with_label = reverse_training_list(val_list)
	
	
	
	print(data.shape)
	
	cross_validation(train_list_with_label,val_list_with_label,test_list,data)




main()
