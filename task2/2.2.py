import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn import preprocessing
import math
import csv
from sklearn.feature_selection import SelectKBest, SelectFpr, SelectFdr, SelectFwe, f_classif, chi2, f_regression,SelectFromModel
from sklearn.svm import SVR, SVC, LinearSVC
from scipy import stats

VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
sep = 'LABEL_Sepsis'
num_feature = 60
Alpha = 0.01
         
def data_process(train_path, test_path, label_path):
	train = load_data(train_path)
	test = load_data(test_path)
	label = load_data(label_path)
	
	train = feature_augment(train);
	print('training data size after feature augmentation:'); print(train.shape)
	test = feature_augment(test);
	print('testing data size after feature augmentation:'); print(test.shape)
	
	train = train.fillna(0);
	test = test.fillna(0);
	# if we should do normalization? if we set the nan to zero.
	
	return train, test, label


def data_processnorm(train_path, test_path, label_path):
	train = load_data(train_path)
	test = load_data(test_path)
	label = load_data(label_path)
	
	train = feature_augment(train);
	print('training data size after feature augmentation:'); print(train.shape)
	test = feature_augment(test);
	print('testing data size after feature augmentation:'); print(test.shape)
   
	train = train.fillna(0);
	test = test.fillna(0); 
    
    # MinMax normalization
	min_max_scaler = preprocessing.MinMaxScaler()
	for indextrain in train.columns:
		train[indextrain] = min_max_scaler.fit_transform(train[indextrain])
	min_max_scaler2 = preprocessing.MinMaxScaler()
	for indextest in train.columns:
		test[indextest] = min_max_scaler2.fit_transform(test[indextrain])
	# if we should do normalization? if we set the nan to zero.
	
	return train, test, label
	
	
def load_data(data_path):
	data = pd.read_csv(data_path)
	#X_train = train.drop(['Id','y'],axis=1)
	#y_train = train.y
	#X_test = test.drop(['Id'],axis=1)
	return data

def feature_augment(data):
	mean_data = data.groupby('pid').mean();
	max_data = data.groupby('pid').max();
	min_data = data.groupby('pid').min();
	median_data = data.groupby('pid').median();
	data = pd.concat([mean_data,max_data,min_data, median_data],axis=1)
	return data
	
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
	
def feature_selectionKbest(data,y,num_feature):
    xx = data.sort_values('pid').values;
    xx_label = y.sort_values('pid')[sep].values;
    select = SelectKBest(f_classif, k=num_feature).fit(xx,xx_label)
    #select = SelectKBest(chi2, k=num_feature).fit(xx,xx_label)
	#select = SelectFromModel(estimator=Lasso(), threshold=-np.inf, max_features=num_feature).fit(data,y)
    reduced_xx = select.transform(xx)
    new_data = select.inverse_transform(reduced_xx)
    new_data = pd.DataFrame(new_data,index = data.sort_values('pid').index, columns = data.sort_values('pid').columns)
	#idx = select.get_support()
	#print(idx)
	#new_data = np.delete(new_data,idx,1)
    return new_data

def feature_Univarselection(data,y,Alpha):
    xx = data.sort_values('pid').values;
    xx_label = y.sort_values('pid')[sep].values;
    select = SelectFpr(f_classif, alpha=Alpha).fit(xx,xx_label)
    #select = SelectFdr(f_classif, alpha=Alpha).fit(xx,xx_label)
    #select = SelectFwe(f_classif, alpha=Alpha).fit(xx,xx_label)
    #select = SelectKBest(chi2, k=num_feature).fit(xx,xx_label)
	#select = SelectFromModel(estimator=Lasso(), threshold=-np.inf, max_features=num_feature).fit(data,y)
    reduced_xx = select.transform(xx)
    new_data = select.inverse_transform(reduced_xx)
    new_data = pd.DataFrame(new_data,index = data.sort_values('pid').index, columns = data.sort_values('pid').columns)
	#idx = select.get_support()
	#print(idx)
	#new_data = np.delete(new_data,idx,1)
    return new_data

def feature_selectionfrommodel(data,y,num_feature):
    xx = data.sort_values('pid').values;
    xx_label = y.sort_values('pid')[sep].values;
    #select = SelectKBest(f_classif, k=num_feature).fit(xx,xx_label)
    #select = SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=10000), threshold= "median", max_features=num_feature).fit(xx,xx_label)
    select = SelectFromModel(RandomForestClassifier(n_estimators=20000, random_state=0, n_jobs=-1),threshold= "median", max_features=num_feature).fit(xx,xx_label)
    reduced_xx = select.transform(xx)
    new_data = select.inverse_transform(reduced_xx)
    new_data = pd.DataFrame(new_data,index = data.sort_values('pid').index, columns = data.sort_values('pid').columns)
	#idx = select.get_support()
	#print(idx)
	#new_data = np.delete(new_data,idx,1)
    return new_data

	
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

def svc(X,y):
	clf = SVC(probability=True, class_weight='balanced');
	clf.fit(X,y);
	return clf;

def cross_validation(data, Y_train, kfold):
	score = 0
	score_train = 0
	kfold = 2
	kf = KFold(n_splits = kfold)
	alpha = 10
	weight = 0
	m,n = data.shape
	for train_index, val_index in kf.split(data):
		x_train, x_val= data[train_index], data[val_index]
		y_train, y_val = Y_train[train_index], Y_train[val_index]
		reg = svc(x_train, y_train)
		y_val_pred = reg.predict_proba(x_val) # shape: (n_sample, n_class)
		score += roc_auc_score(y_val, y_val_pred[:,1]) * len(y_val)
		#score_train += (mean_squared_error(reg.predict(x_train), y_train)) * len(y_train)
		#weight += reg.coef_
	#print("{}-fold cross validation RMSE: {} with {} features".format(kfold, math.sqrt(score / len(Y_train)), n))
	#print("{}-fold training RMSE: {} ".format(kfold, math.sqrt(score_train/ (len(Y_train)*kfold))))
	return (score / len(Y_train))

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
	result.to_csv('./weight.csv',indata_processdex=False,header=False)
	
def do_task1(train, label_data, test):
	for label in TESTS:
		print(label)
		x_data = train.sort_values('pid').values;
		x_label = label_data.sort_values('pid')[label].values;
		score = cross_validation(x_data, x_label, 20);
		print("score of {}:{}".format(label, score));
	return

def do_task2(train, label_data, test):
	print(sep)
	x_data = train.sort_values('pid').values;
	x_label = label_data.sort_values('pid')[sep].values;
	score = cross_validation(x_data, x_label, 20);
	print("score of {}:{}".format(sep, score));
	return		

def main():
	train_path = './train_features.csv';
	test_path = './test_features.csv';
	label_path = './train_labels.csv';
	#train, test, label = data_process(train_path, test_path, label_path); # still return pandaFrame
	#do_task1(train, label, test)
    # if need values, just use 'train.values' it will return numpy array, label['LABEL_ABPm'].values to return labels.
	# task 1do_task1()
    # task 2	
	train, test, label = data_processnorm(train_path, test_path, label_path); # still return pandaFrame
	#new_train = feature_selectionKbest(train,label,num_feature)
	new_train = feature_Univarselection(train,label,Alpha)
	#new_train = feature_selectionfrommodel(train,label,num_feature)
	do_task2(new_train, label, test)
    # task 3
    
	#kfold = 20
	#loss = np.inf
	#weight_ = np.zeros(21)
	#for num_feature in range(10,15):
#		data_, idx = feature_selection(data, Y_train,num_feature)
#		weight , val_loss = cross_validation(data_, Y_train, kfold)
#		if val_loss<loss: 
#			loss = val_loss; 
#			weight_ = weight; 
#			best_num = num_feature;
#			idx_ = idx;
		
	#data, idx = feature_selection(data, Y_train,num_feature)
main()
		
		
