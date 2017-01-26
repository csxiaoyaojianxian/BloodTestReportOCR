# -*- coding: utf-8 -*-
import numpy as np
import random
import subprocess
import platform
import sys,os
sys.path.append('/home/summer/Desktop/caffe/python')
import caffe
import lmdb
from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def extract(filename):
    matrix = np.loadtxt(filename, dtype='string', skiprows= 1,delimiter=',', usecols=(1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28))
    matrix = matrix_filter(matrix)
    matrix = np.asarray(matrix)

    data = matrix[:,1:27]
    sex  = matrix[:,0]
    data = data.astype(np.float) #convert string to float
    for i in range(len(sex)):
        if sex[i] == '\xc4\xd0':
            sex[i] = 1
        else :
        	if sex[i] != '1':
           		sex[i] = 0
    return data,sex
#filter the row which contains the wrong elements
def matrix_filter(matrix):
    count = 0
    flag = 0
    for row in matrix:
        for cloumn in row:
            if cloumn == '--.--' or cloumn == '':  #Discard the wrong value
                flag = 1

        if count == 0 and flag == 0:
            table = row
            count = 1
            continue
        if flag == 0 :
            table = np.c_[table,row]   #Add the elements by extend cloumns
        else:
            flag = 0
    table = table.transpose()          #Transpose the matrix
    return table

#nomalize the data
def nomalize(X_train, X_test):

    ave = X_train.mean(axis=0) # get the average of cloumns
    std = X_train.std(axis=0)  # get the standard deviation of cloumns
    train_table = [(row - ave)/std for row in X_train]
    X_train = (np.asarray(train_table))

    test_table = [(row - ave)/std for row in X_test]
    X_test = (np.asarray(test_table))
    return X_train, X_test

#load data into lmdb
def load_data_into_lmdb(lmdb_name, features, labels=None):
    env = lmdb.open(lmdb_name, map_size=features.nbytes*10)

    features = features[:,:,None,None]
    for i in range(features.shape[0]):
        datum = caffe.proto.caffe_pb2.Datum()

        datum.channels = features.shape[1]   # features's number(26)
        datum.height = 1                     # due to eachone only have one data
        datum.width = 1                      # so the size is 1x1

        if features.dtype == np.int:         # convert data to string
            datum.data = features[i].tostring()
        elif features.dtype == np.float:
            datum.float_data.extend(features[i].flat)
        else:
            raise Exception("features.dtype unknown.")

        if labels is not None:
            datum.label = int(labels[i])

        str_id = '{:08}'.format(i)
        with env.begin(write=True) as txn:
            txn.put(str_id, datum.SerializeToString())

def get_data_from_lmdb_evalue(lmdb_name):
    lmdb_env = lmdb.open(lmdb_name, readonly=True)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    success = 0
    count = 0
    #raw_datum = lmdb_txn.get()
    for key, value in lmdb_cursor:

        datum.ParseFromString(value)
        label = datum.label
        feature = caffe.io.datum_to_array(datum)
        out = net.forward(**{net.inputs[0]: np.asarray([feature])})
        count+=1
        if np.argmax(out["prob"][0]) == label :
            success+=1
            print "success", out
    return count,success

def create_data_lmdb():

    #prefit
    X, y = extract('data_set.csv')
    vec_log = np.vectorize(lambda x: x)
    vec_int = np.vectorize(lambda str: int(str[-1]))
    features = vec_log(X)
    labels = vec_int(y)

    #train : test = 9 : 1
    sss = StratifiedShuffleSplit(labels, 1, test_size=0.1, random_state=0)
    sss = list(sss)[0]

    features_training = features[sss[0],]
    labels_training = labels[sss[0],]

    features_testing = features[sss[1],]
    labels_testing = labels[sss[1],]

    #nomalized data 66%, unnomalized data 53%
    features_training, features_testing = nomalize(features_training, features_testing)

    load_data_into_lmdb("train_data_lmdb", features_training, labels_training)
    load_data_into_lmdb("test_data_lmdb", features_testing, labels_testing)

if __name__=='__main__':
    #建立lmdb格式数据库，只需创建一次，再次创建需要清除原来数据文件
    create_data_lmdb();
    #根据配置文件开始训练模型
    solver = caffe.get_solver("config.prototxt")
    solver.solve()

    net = caffe.Net("model_prod_prototxt","_iter_500000.caffemodel", caffe.TEST)

    # if the index of the largest element matches the integer
    # label we stored for that case - then the prediction is right
    total,success = get_data_from_lmdb_evalue("test_data_lmdb/")
    print "accuracy:", success*100/total,"%"
