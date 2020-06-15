import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, RidgeCV,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score, f1_score
from sklearn import preprocessing
import math
import csv
from xgboost import XGBRegressor
import random

from sklearn.feature_selection import SelectKBest, SelectFpr, SelectFdr, SelectFwe, f_classif, chi2, f_regression,SelectFromModel
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from statistics import mean
from data import DataSet

num_feature = 60
Alpha = 0.01
Amino_acids = np.array(['R','H','K','D','E','S','T','N','Q','C','U','G','P','A','I','L','M','F','W','Y','V'])


def encoder_sparse(train_path, test_path):
    train = load_data(train_path)
    test = load_data(test_path)
    n_train,m_train = train.shape
    n_test,m_test = test.shape
    new_train = np.zeros(n_train*84).reshape(n_train,84)    
    new_test = np.zeros(n_test*84).reshape(n_test,84)
    Seq_train = train['Sequence'].values
    Seq_test = test['Sequence'].values
    label = train['Active'].values
    for train_idx in range(n_train):
        seq_train_arr = np.array(list(Seq_train[train_idx]))
        for i in range(4):
            seq_train_idx = int(np.where(Amino_acids == seq_train_arr[i])[0])
            new_train[train_idx,i*21+seq_train_idx] = 1
    
    for test_idx in range(n_test):
        seq_test_arr = np.array(list(Seq_test[test_idx]))
        for j in range(4):
            seq_test_idx = int(np.where(Amino_acids == seq_test_arr[j])[0])
            new_test[test_idx,j*21+seq_test_idx] = 1
            

            
    return new_train, new_test, label

def encoder(train_path, test_path):
    train = load_data(train_path)
    test = load_data(test_path)
    n_train,m_train = train.shape
    
    n_test,m_test = test.shape
    new_train = np.zeros(n_train*4).reshape(n_train,4)    
    new_test = np.zeros(n_test*4).reshape(n_test,4)
    Seq_train = train['Sequence'].values
    Seq_test = test['Sequence'].values
    label = train['Active'].values
    for train_idx in range(n_train):
        seq_train_arr = np.array(list(Seq_train[train_idx]))
        for i in range(4):
            seq_train_idx = int(np.where(Amino_acids == seq_train_arr[i])[0])
            new_train[train_idx,i] = seq_train_idx
    
    for test_idx in range(n_test):
        seq_test_arr = np.array(list(Seq_test[test_idx]))
        for j in range(4):
            seq_test_idx = int(np.where(Amino_acids == seq_test_arr[j])[0])
            new_test[test_idx,j] = seq_test_idx
    
    # if we should do normalization? if we set the nan to zero.

    return new_train, new_test, label

def data_process(train_path, test_path):
    train = load_data(train_path)
    test = load_data(test_path)

    # if we should do normalization? if we set the nan to zero.

    return train, test


    


	
def cross_validation(data, Y_train, test):
    score = 0
    score_train = 0
    kfold = 10
    kf = KFold(n_splits=kfold)
    alpha = 10
    weight = 0
    m, n = data.shape
    for train_index, val_index in kf.split(data):
        x_train, x_val = data[train_index], data[val_index]
        y_train, y_val = Y_train[train_index], Y_train[val_index]
        #reg = svc(x_train, y_train)
        reg = xgb(x_train, y_train)        
        #reg = logistic(x_train,y_train)
        # y_val_pred = reg.predict_proba(x_val) # shape: (n_sample, n_class)
        y_val_pred = reg.predict(x_val)
        # score += roc_auc_score(y_val, y_val_pred[:,1]) * len(y_val)
        score += f1_score(y_val, y_val_pred) * len(y_val)
        
        pred = reg.predict(test);
    # score_train += (mean_squared_error(reg.predict(x_train), y_train)) * len(y_train)
    # weight += reg.coef_
    return (score / len(Y_train)), np.array(pred)



    
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


def do_task(train, label_data, test):

    # do feature selection before training
    #x_data, test_selected = feature_selection(x_data, x_label, 70,test.sort_values('pid').values);
    score, pred = cross_validation(train, label_data, test);

    print("f1_score :{}".format(score));
    result = pd.DataFrame(pred)
    result.to_csv('./sample.csv',index=False,header=False)
    return

def main():
    train_path = './train_triplets.txt';
    test_path = './test_triplets.txt';
    data_saved = True
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


    data = DataSet(train_list,test_list)
    if data_saved:
        buffer = np.load('buffer.npy')
    else:
        buffer = data.compress_data_to_numpy()


main()
