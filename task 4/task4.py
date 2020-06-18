# import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
import argparse
from Model import Model
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from sklearn.feature_selection import SelectKBest, SelectFpr, SelectFdr, SelectFwe, f_classif, chi2, f_regression, \
    SelectFromModel
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from statistics import mean
from data import DataSet
# from alex import triplet_loss,get_alex,triplet_loss_np
num_feature = 60
Alpha = 0.01
Amino_acids = np.array(
    ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V'])



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
    # x_data, test_selected = feature_selection(x_data, x_label, 70,test.sort_values('pid').values);
    score, pred = cross_validation(train, label_data, test);

    print("f1_score :{}".format(score));
    result = pd.DataFrame(pred)
    result.to_csv('./sample.csv', index=False, header=False)
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

    data = DataSet(train_list, test_list)

    # train,val = data.split_train_val(data,0.9)
    if data_saved:
        buffer = np.load('buffer.npy')
    else:
        buffer = data.compress_data_to_numpy()
    # train, test = data.get_data()
    # --- 训练部分
    # model = get_alex(resize = 28)
    # model.compile(optimizer=tf.optimizers.Adam(0.01),
    #               loss=triplet_loss_np,  # mean squared error
    #               metrics=['accuracy'])
    # model.fit(data, epochs=10, steps_per_epoch=30,
    #           validation_data=test,
    #           validation_steps=3)

    boolean = lambda x: bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--Ischeckpoint', default=False, type=boolean, help='If load the saved model')
    parser.add_argument('--nb_epochs', default=20, type=int, help='epochs')

    opt = parser.parse_args()

    model = Model(opt)

    # train_list_ = random.sample(train_list,int(len(train_list)*0.1))
    train_list_, val_list_ = data.split_train_val(train_list,0.8)

    for epoch in range(opt.nb_epochs):


        temp = random.sample(train_list_,len(train_list_))
        mini_batches = [temp[k:k+opt.batch_size] for k in range(0,len(temp)-opt.batch_size,opt.batch_size)]
        
        for iteration, mini_batch in enumerate(mini_batches):

            Image = data.get_batch(buffer,mini_batch);  # batchsize *3 * 28 * 28* 3
            loss,positive_distance,negative_distance = model.update(Image)

            if iteration % 50 == 0:
                # print(positive_distance- negative_distance)
                print("epoch %d : batch %d: loss %f" % (epoch, iteration, np.mean(loss)))

        # for validation use

        temp_val = random.sample(val_list_,len(val_list_))
        mini_batches = [temp_val[k:k+opt.batch_size] for k in range(0,len(temp_val)-opt.batch_size,opt.batch_size)]

        total_num = 0
        test_loss = 0
        for iteration,mini_batch in enumerate(mini_batches):

            Image = data.get_batch(buffer,mini_batch)
            loss,positive_distance,negative_distance = model.test(Image)

            test_loss += loss

            if iteration % 50 == 0:
                print("epoch %d : batch %d: validation loss %f" % (epoch, iteration, np.mean(loss)))


            for i in range(opt.batch_size):
                total_num += 1
                if positive_distance<negative_distance:
                    corr_num += 1
        acc = corr_num / total_num
        print('validation accuracy: ', acc, 'total_loss', test_loss)



main()