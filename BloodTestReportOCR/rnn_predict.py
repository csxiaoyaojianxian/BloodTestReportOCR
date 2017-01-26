# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

def predict_sex(data_predict):
    tf.reset_default_graph()

    # Network Parameters
    n_input = 11  # MNIST data input (img shape: 28*28)
    n_steps = 2  # timesteps
    n_hidden = 128  # hidden layer num of features
    n_classes = 2  # MNIST total classes (0-9 digits)

    data_predict = np.reshape(data_predict, (1,n_steps, n_input))




    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def RNN(x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)


        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        #to : (n_steps, batch_size, n_input)


        # Dimensionality reduction
        x = tf.reshape(x, [-1, n_input])
        # Reshaping to (n_steps*batch_size, n_input)

        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    pred = RNN(x, weights, biases)

    # Initializing the variables
    init = tf.global_variables_initializer()

    ######
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess,"./rnn_model/rnn_sex_model/model.ckpt")
        p = sess.run(pred, feed_dict={x:data_predict})



    if p[0][0] > p[0][1]:
        sex_result = 0
    else:
        sex_result = 1


    return sex_result



def predict_age(data_predict):
    tf.reset_default_graph()

    # Network Parameters
    n_input = 11  # MNIST data input (img shape: 28*28)
    n_steps = 2  # timesteps
    n_hidden = 128  # hidden layer num of features
    n_classes = 10 #MNIST total classes (0-9 digits)

    data_predict = np.reshape(data_predict, (1,n_steps, n_input))




    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def RNN(x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)


        # Permuting batch_size and n_steps

        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    pred = RNN(x, weights, biases)

    # Initializing the variables
    init = tf.global_variables_initializer()

    ######
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess,"./rnn_model/rnn_age_model/model.ckpt")
        p = sess.run(pred, feed_dict={x:data_predict})

    # print(tf.argmax(p, 1))
    max = p[0][0]
    max_i = 0
    for i in range(n_classes):
        if p[0][i] > max:
            max_i = i
            max = p[0][i]


    age_result = str(max_i * 10) + "~" + str((max_i+1) *10 -1)

    return age_result