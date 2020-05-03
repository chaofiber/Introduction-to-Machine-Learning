import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score
from sklearn import preprocessing
import math
import csv
from xgboost import XGBRegressor
import random

from sklearn.feature_selection import SelectKBest, SelectFpr, SelectFdr, SelectFwe, f_classif, chi2, f_regression,SelectFromModel
from sklearn.svm import SVR, SVC, LinearSVC
from scipy import stats
from statistics import mean 



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
    print('training data size after feature augmentation:');
    print(train.shape)
    test = feature_augment(test);
    print('testing data size after feature augmentation:');
    print(test.shape)

    train = train.fillna(0);
    test = test.fillna(0);
    # if we should do normalization? if we set the nan to zero.

    return train, test, label


def data_processnorm(train_path, test_path, label_path):
    train = load_data(train_path)
    test = load_data(test_path)
    label = load_data(label_path)

    train = feature_augment(train);
    print('training data size after feature augmentation:');
    print(train.shape)
    test = feature_augment(test);
    print('testing data size after feature augmentation:');
    print(test.shape)

    # MinMax normalization
    # if we should do normalization? if we set the nan to zero.

    return train, test, label

def data_processnorm2(train_path, test_path, label_path):
    train = load_data(train_path)
    test = load_data(test_path)
    label = load_data(label_path)

    train = feature_augment2(train);
    print('training data size after feature augmentation:');
    print(train.shape)
    test = feature_augment2(test);
    print('testing data size after feature augmentation:');
    print(test.shape)

    # MinMax normalization
    # if we should do normalization? if we set the nan to zero.

    return train, test, label

def data_process_mean(train_path,test_path,label_path):
    train = load_data(train_path)
    test = load_data(test_path)
    label = load_data(label_path)
    train = feature_augment(train);
    print('training data size after feature augmentation:');
    print(train.shape)
    test = feature_augment(test);
    print('testing data size after feature augmentation:');
    print(test.shape)

    train = train.interpolate('pad')
    test = test.interpolate('pad')
    return train,test,label

def load_data(data_path):
    data = pd.read_csv(data_path)
    # X_train = train.drop(['Id','y'],axis=1)
    # y_train = train.y
    # X_test = test.drop(['Id'],axis=1)
    return data
def regression(X,y):
    xg_reg = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.05,
	max_depth = 5, alpha = 10, n_estimators = 160)
    xg_reg.fit(X,y)
    return xg_reg

def feature_augment(data):
    mean_data = data.groupby('pid').mean();
    max_data = data.groupby('pid').max();
    min_data = data.groupby('pid').min();
    median_data = data.groupby('pid').median();
    data = pd.concat([mean_data, max_data, min_data, median_data], axis=1)
    return data

def feature_augment2(data):
    mean_data = data.groupby('pid').mean();
    max_data = data.groupby('pid').max();
    min_data = data.groupby('pid').min();
    median_data = data.groupby('pid').median();
    std_data = data.groupby('pid').std();
    quan1_data = data.groupby('pid').quantile(.25);
    quan2_data = data.groupby('pid').quantile(.75);   
    count_data = data.groupby('pid').count();
    data = pd.concat([mean_data, max_data, min_data, median_data,std_data,quan1_data,quan2_data,count_data], axis=1)
    return data


def feature_transform(data):
    n, m = data.shape
    new_data = np.zeros(n * 21).reshape(n, 21)
    new_data[:, 0:5] = data[:, 0:5]
    new_data[:, 5:10] = np.power(data[:, 0:5], 2);  # element wise multiplication
    print(new_data[1, 5:10])
    print(data[1, 0:5])
    new_data[:, 10:15] = np.exp(data[:, 0:5])
    new_data[:, 15:20] = np.cos(data[:, 0:5])
    new_data[:, 20] = 1
    # order: 1,11,10
    # new_data = np.delete(new_data,idx,1)
    return new_data


def feature_selectionKbest(data, y, num_feature):
    xx = data.sort_values('pid').values;
    xx_label = y.sort_values('pid')[sep].values;
    select = SelectKBest(f_classif, k=num_feature).fit(xx, xx_label)
    # select = SelectKBest(chi2, k=num_feature).fit(xx,xx_label)
    # select = SelectFromModel(estimator=Lasso(), threshold=-np.inf, max_features=num_feature).fit(data,y)
    reduced_xx = select.transform(xx)
    new_data = select.inverse_transform(reduced_xx)
    new_data = pd.DataFrame(new_data, index=data.sort_values('pid').index, columns=data.sort_values('pid').columns)
    # idx = select.get_support()
    # print(idx)
    # new_data = np.delete(new_data,idx,1)
    return new_data


def feature_Univarselection(data, y, Alpha):
    xx = data.sort_values('pid').values;
    xx_label = y.sort_values('pid')[sep].values;
    select = SelectFpr(f_classif, alpha=Alpha).fit(xx, xx_label)
    # select = SelectFdr(f_classif, alpha=Alpha).fit(xx,xx_label)
    # select = SelectFwe(f_classif, alpha=Alpha).fit(xx,xx_label)
    # select = SelectKBest(chi2, k=num_feature).fit(xx,xx_label)
    # select = SelectFromModel(estimator=Lasso(), threshold=-np.inf, max_features=num_feature).fit(data,y)
    reduced_xx = select.transform(xx)
    new_data = select.inverse_transform(reduced_xx)
    new_data = pd.DataFrame(new_data, index=data.sort_values('pid').index, columns=data.sort_values('pid').columns)
    # idx = select.get_support()
    # print(idx)
    # new_data = np.delete(new_data,idx,1)
    return new_data


def feature_selectionfrommodel(data, y, num_feature):
    xx = data.sort_values('pid').values;
    xx_label = y.sort_values('pid')[sep].values;
    # select = SelectKBest(f_classif, k=num_feature).fit(xx,xx_label)
    # select = SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=10000), threshold= "median", max_features=num_feature).fit(xx,xx_label)
    select = SelectFromModel(RandomForestClassifier(n_estimators=20000, random_state=0, n_jobs=-1), threshold="median",
                             max_features=num_feature).fit(xx, xx_label)
    reduced_xx = select.transform(xx)
    new_data = select.inverse_transform(reduced_xx)
    new_data = pd.DataFrame(new_data, index=data.sort_values('pid').index, columns=data.sort_values('pid').columns)
    # idx = select.get_support()
    # print(idx)
    # new_data = np.delete(new_data,idx,1)
    return new_data


def feature_selection(data,y,num_feature,test):
	select = SelectKBest(f_regression, k=num_feature).fit(data,y)
	#select = SelectFromModel(estimator=Lasso(), threshold=-np.inf, max_features=num_feature).fit(data,y)
	new_data = select.transform(data);
	new_test = select.transform(test);
	idx = select.get_support()
	#print(idx)
	#new_data = np.delete(new_data,idx,1)
	return new_data, new_test
	
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


def xgb(X,y):
	xg_reg = XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.3,
	max_depth = 5, alpha = 10, n_estimators = 100);
	xg_reg.fit(X,y);
	return xg_reg;

def xgb2(X,y):
	xg_reg = XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.31, learning_rate = 0.05,
	max_depth =5 , alpha = 10,n_estimators = 121);
	xg_reg.fit(X,y);
	return xg_reg;


def logistic(X,y):
	clf = LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced',max_iter=500).fit(X, y)
	return clf;
	
def cross_validation(data, Y_train, test):
    score = 0
    score_train = 0
    kfold = 10
    kf = KFold(n_splits=kfold)
    alpha = 10
    weight = 0
    m, n = data.shape
    pred = []
    for train_index, val_index in kf.split(data):
        x_train, x_val = data[train_index], data[val_index]
        y_train, y_val = Y_train[train_index], Y_train[val_index]
        # reg = svc(x_train, y_train)
        reg = xgb(x_train, y_train)
        #reg = logistic(x_train,y_train)
        # y_val_pred = reg.predict_proba(x_val) # shape: (n_sample, n_class)
        y_val_pred = reg.predict(x_val)
        # score += roc_auc_score(y_val, y_val_pred[:,1]) * len(y_val)
        score += roc_auc_score(y_val, y_val_pred) * len(y_val)
        
        pred.append(reg.predict(test));
    # score_train += (mean_squared_error(reg.predict(x_train), y_train)) * len(y_train)
    # weight += reg.coef_
    return (score / len(Y_train)), np.mean(np.array(pred),0)

def cross_validation_task2(data, Y_train, test):

    score = 0
    score_train = 0
    kfold = 5
    kf = KFold(n_splits=kfold)
    alpha = 10
    weight = 0
    m, n = data.shape
    pred = []
    for train_index, val_index in kf.split(data):
        x_train, x_val = data[train_index], data[val_index]
        y_train, y_val = Y_train[train_index], Y_train[val_index]
        # reg = svc(x_train, y_train)
        reg = xgb2(x_train, y_train)
        # y_val_pred = reg.predict_proba(x_val) # shape: (n_sample, n_class)
        y_val_pred = reg.predict(x_val)
        # score += roc_auc_score(y_val, y_val_pred[:,1]) * len(y_val)
        score += roc_auc_score(y_val, y_val_pred) * len(y_val)
        
        pred.append(reg.predict(test));
    # score_train += (mean_squared_error(reg.predict(x_train), y_train)) * len(y_train)
    # weight += reg.coef_
    return (score / len(Y_train)), np.mean(np.array(pred),0)

def cross_validation_reg(data, Y_train, test):
    score = 0
    score_train = 0
    kfold = 5
    kf = KFold(n_splits = kfold)
    weight = 0
    m,n = data.shape
    pred = [];
    for train_index, val_index in kf.split(data):
        x_train, x_val= data[train_index], data[val_index]
        y_train, y_val = Y_train[train_index], Y_train[val_index]
        # print(len(y_val), len(Y_train))
        reg = regression(x_train, y_train)
        y_val_pred = reg.predict(x_val) # shape: (n_sample, n_class)
        print(y_val_pred)
        pred.append(reg.predict(test));

        score += (0.5 + 0.5 * np.maximum(0, r2_score(y_val, y_val_pred)))*len(y_val)

    return (score / len(Y_train)), np.mean(np.array(pred),0)
    
def print_to_csv(weight, idx):
    j = 0;
    weight_ = np.zeros(21);
    for i in range(21):
        # if i in idx:
        if idx[i] == False:
            weight_[i] = 0
        else:
            weight_[i] = weight[j]
            j += 1
    result = pd.DataFrame(weight_)
    result.to_csv('./weight.csv', indata_processdex=False, header=False)


def do_task1(train, label_data, test):
    total_score = [];
    submit = pd.DataFrame()
    test_id = test.sort_values('pid').index
    submit['pid'] = test_id
    for label in TESTS:
        print(label)
        x_data = train.sort_values('pid').values;
        x_label = label_data.sort_values('pid')[label].values;

        # do feature selection before training
        #x_data, test_selected = feature_selection(x_data, x_label, 70,test.sort_values('pid').values);
        score, pred = cross_validation(x_data, x_label, test.sort_values('pid').values);

        submit[label] = pred;
        
        total_score.append(score);
        print("score of {}:{}".format(label, score));
    print("average score of subtask1:{}".format(mean(total_score)));
    submit.to_csv('submission.csv',index=False)
    return


def do_task2(train, label_data, test):
    print(sep)
    submit = pd.read_csv('submission.csv');
    x_data = train.sort_values('pid').values;
    x_label = label_data.sort_values('pid')[sep].values;
    score, pred = cross_validation_task2(x_data, x_label, test.sort_values('pid').values);
    # submit[sep] = random.random()
    submit[sep] = pred;
    submit.to_csv('submission.csv',index=False)
    print("score of {}:{}".format(sep, score));
    return

def do_task3(train, label_data, test):
    mean_score = 0
    submit = pd.read_csv('submission.csv');
    for label in VITALS:
        # print(label)
        # --- Do regression tasks
        x_data = train.sort_values('pid').values
        x_label = label_data.sort_values('pid')[label].values

        score, pred = cross_validation_reg(x_data, x_label, test.sort_values('pid').values)
        submit[label] = pred
        # submit[label] = 100*random.random()
        print("score of {}:{}".format(label, score))
        mean_score += score
    print("mean score: {}".format(mean_score/len(VITALS)))
    #submit.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')
    submit.to_csv('submission.csv',index=False)

    return
def main():

    train_path = './train_features.csv';
    test_path = './test_features.csv';
    label_path = './train_labels.csv';
    train, test, label = data_processnorm(train_path, test_path, label_path);  # still return pandaFrame
    #train, test, label = data_process_mean(train_path, test_path, label_path)
    # if need values, just use 'train.values' it will return numpy array, label['LABEL_ABPm'].values to return labels.
    # task 1
    print("starting subtask1");
    do_task1(train, label, test)
    print("finish subtask1");
    # task 2
    # train, test, label = data_processnorm(train_path, test_path, label_path);  # still return pandaFrame
    # # new_train = feature_selectionKbest(train,label,num_feature)
    # new_train = feature_Univarselection(train, label, Alpha)
    # new_train = feature_selectionfrommodel(train,label,num_feature)
    print("starting subtask2");
    train, test, label = data_processnorm2(train_path, test_path, label_path);  
    do_task2(train, label, test)
    print("finish subtask2");
    

# task 3


    train, test, label = data_process_mean(train_path, test_path, label_path)
    # new_train = feature_Univarselection(train, label, Alpha)
    print("starting subtask3");
    do_task3(train,label,test)
    print("finish subtask3");
    


main()
