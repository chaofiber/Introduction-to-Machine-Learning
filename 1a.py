import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import math
import csv


def load_data(train_path):
	train = pd.read_csv(train_path)
	#test = pd.read_csv(test_path)
	X_train = train.drop(['Id','y'],axis=1)
	y_train = train.y
	#X_test = test.drop(['Id'],axis=1)
	return X_train.values, y_train.values
	
def main():
	train_path = './train.csv';
	X_train, Y_train = load_data(train_path);
	# K-Fold is used, default: k=5
	# kf = KFold(n_splits = 10)
	# score = 0
	alpha = 0.01
	#print(X_train.shape)
	#print(Y_train.shape)
	number = 0
	for i in range(5):
		# K-Fold is used, default: k=5
		score = 0
		kf = KFold(n_splits = 10)
		for train_index, val_index in kf.split(X_train):
			x_train, x_val= X_train[train_index], X_train[val_index]
			y_train, y_val = Y_train[train_index], Y_train[val_index]
			ridge = Ridge(alpha=alpha);
			ridge.fit(x_train,y_train)
			y_val_pred = ridge.predict(x_val)
			score += mean_squared_error(y_val_pred,y_val) * len(y_val)
			#number += len(y_val)
		if i==0:
			rmse = [math.sqrt(score / len(X_train))]
		else:
			rmse.append(math.sqrt(score / len(X_train)))
		print("{}-fold cross validation RMSE: {} with alpha={}".format(10, math.sqrt(score / len(X_train)), alpha))
		alpha *= 10
		#print(number, len(X_train))
	# Obtain the final classifier on all data
	# reg = LinearRegression().fit(X_train, Y_train)
	# y_submit = reg.predict(X_test)
	# result = pd.DataFrame()
	# result['Id'] = testId
	#print(rmse)
	result = pd.DataFrame(rmse)
	print(result)
	result.to_csv('./rmse.csv',index=False,header=False)
	#with open('./rmse.csv','w',newline='') as myfile:
	#	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	#	for value in rmse:
	#		wr.writerow([value])
	

main()
		
		
