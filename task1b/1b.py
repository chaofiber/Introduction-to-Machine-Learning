import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import math
import csv
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression,SelectFromModel
from sklearn.svm import SVR
from scipy import stats

def load_data(train_path):
	train = pd.read_csv(train_path)
	#test = pd.read_csv(test_path)
	X_train = train.drop(['Id','y'],axis=1)
	y_train = train.y
	#X_test = test.drop(['Id'],axis=1)
	return X_train.values, y_train.values
	
def feature_transform(data):
	n,m = data.shape
	new_data = np.zeros(n*21).reshape(n,21)
	new_data[:,0:5] = data[:,0:5]
	new_data[:,5:10] = np.power(data[:,0:5],2); #element wise multiplication
	print(new_data[1,5:10])
	print(data[1,0:5])
	new_data[:,10:15] = np.exp(data[:,0:5])
	new_data[:,15:20] = np.cos(data[:,0:5])
	new_data[:,20] = 1
	# order: 1,11,10
	#new_data = np.delete(new_data,idx,1)
	return new_data
	
def feature_selection(data,y,num_feature):
	select = SelectKBest(f_regression, k=num_feature).fit(data,y)
	#select = SelectFromModel(estimator=Lasso(), threshold=-np.inf, max_features=num_feature).fit(data,y)
	new_data = select.transform(data);
	idx = select.get_support()
	print(idx)
	#new_data = np.delete(new_data,idx,1)
	return new_data, idx
	
def feature_selection_by_corre(data,y):
	for i in range(21):
		print(stats.pearsonr(data[:,i],y))
	return data
	
def ridgecv(X,y):
	reg = RidgeCV(alphas=[1e-1,1,10,100], fit_intercept=False, cv=30).fit(X,y)
	return reg

def ridge(X,y,alpha):
	reg = Ridge(alpha=alpha,fit_intercept='False',tol=1e-6,solver='svd');
	reg.fit(X,y)
	return reg

def cross_validation(data, Y_train, kfold):
	score = 0
	score_train = 0
	kfold = 30
	kf = KFold(n_splits = kfold)
	alpha = 10
	weight = 0
	m,n = data.shape
	for train_index, val_index in kf.split(data):
		x_train, x_val= data[train_index], data[val_index]
		y_train, y_val = Y_train[train_index], Y_train[val_index]
		reg = ridge(x_train, y_train,alpha)
		y_val_pred = reg.predict(x_val)
		score += mean_squared_error(y_val_pred,y_val) * len(y_val)
		score_train += (mean_squared_error(reg.predict(x_train), y_train)) * len(y_train)
		weight += reg.coef_
	print("{}-fold cross validation RMSE: {} with {} features".format(kfold, math.sqrt(score / len(Y_train)), n))
	print("{}-fold training RMSE: {} ".format(kfold, math.sqrt(score_train/ (len(Y_train)*kfold))))
	return weight/kfold, math.sqrt(score / len(Y_train))

def print_to_csv(weight,idx):
	j = 0;
	weight_ = np.zeros(21);
	for i in range(21):
		#if i in idx:
		if idx[i] == False:
			weight_[i] = 0
		else:
			weight_[i] = weight[j]
			j += 1
	result = pd.DataFrame(weight_)
	result.to_csv('./weight.csv',index=False,header=False)
	
def main():
	train_path = './train.csv';
	X_train, Y_train = load_data(train_path)
	data = feature_transform(X_train)
	kfold = 20
	loss = np.inf
	weight_ = np.zeros(21)
	for num_feature in range(10,15):
		data_, idx = feature_selection(data, Y_train,num_feature)
		weight , val_loss = cross_validation(data_, Y_train, kfold)
		if val_loss<loss: 
			loss = val_loss; 
			weight_ = weight; 
			best_num = num_feature;
			idx_ = idx;
		
	#data, idx = feature_selection(data, Y_train,num_feature)
	print(loss);print(best_num);print(idx_)
	print_to_csv(weight_,idx_)
	

main()
		
		
