# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def normalized(a,b):
    for i in range(22):
        tmp = np.mean(a[:, i])

        a[:, i] = a[:, i] - tmp
        b[:, i] = b[:, i] - tmp


        if np.min(a[:, i]) != np.max(a[:, i]):
            b[:, i] = 2 * (b[:, i] - np.min(a[:, i])) / (np.max(a[:, i]) - np.min(a[:, i])) - 1
        else:
            b[:, i] = 0
    return b

def predict(data_predict):
    tf.reset_default_graph()
    data_nor = np.loadtxt(open("./data.csv", "rb"), delimiter=",", skiprows=0)

    data_predict = normalized(data_nor[:, 2:], data_predict)

    '''
        参数
        '''
    learning_rate = 0.005
    display_step = 100
    n_input = 22

    n_hidden_1_age = 32
    n_hidden_2_age = 16
    n_classes_age = 1

    n_hidden_1_sex = 16
    n_hidden_2_sex = 8
    n_classes_sex = 2
    data = np.loadtxt(open("./data.csv", "rb"), delimiter=",", skiprows=0)
    '''
    建立年龄模型
    '''
    x_age = tf.placeholder("float", [None, n_input])
    y_age = tf.placeholder("float", [None, n_classes_age])

    def multilayer_perceptron_age(x_age, weights_age, biases_age):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x_age, weights_age['h1']), biases_age['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights_age['h2']), biases_age['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights_age['out']) + biases_age['out']
        return out_layer

    weights_age = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1_age])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1_age, n_hidden_2_age])),
        'out': tf.Variable(tf.random_normal([n_hidden_2_age, n_classes_age]))
    }
    biases_age = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1_age])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2_age])),
        'out': tf.Variable(tf.random_normal([n_classes_age]))
    }
    pred_age = multilayer_perceptron_age(x_age, weights_age, biases_age)
    '''
    建立性别模型
    '''
    x_sex = tf.placeholder("float", [None, n_input])
    y_sex = tf.placeholder("float", [None, n_classes_sex])

    def multilayer_perceptron_sex(x_sex, weights_sex, biases_sex):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x_sex, weights_sex['h1']), biases_sex['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights_sex['h2']), biases_sex['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights_sex['out']) + biases_sex['out']
        return out_layer

    weights_sex = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1_sex])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1_sex, n_hidden_2_sex])),
        'out': tf.Variable(tf.random_normal([n_hidden_2_sex, n_classes_sex]))
    }
    biases_sex = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1_sex])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2_sex])),
        'out': tf.Variable(tf.random_normal([n_classes_sex]))
    }
    pred_sex = multilayer_perceptron_sex(x_sex, weights_sex, biases_sex)

    '''
    共同的初始化
    '''
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        saver.restore(sess, "./nn_model/model.ckpt")
        print ("load model success!")
        p_sex = sess.run(pred_sex, feed_dict={x_sex: data_predict})
        p_age = sess.run(pred_age, feed_dict={x_age: data_predict})
    if p_sex[0][0] > p_sex[0][1]:
        sex_result = 1
    else:
        sex_result = 0

    age_result = p_age[0][0] * 50 +50

    return sex_result,age_result