from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

learning_rate = 0.002
training_iters = 1858
Text_iters = 200
display_step = 10



n_input = 13
n_steps = 2
n_hidden = 64 
n_classes = 2 

def one_hot(a, length):
    b = np.zeros([length, 2])
    for i in range(length):
        if a[i] == 0:
            b[i][1] = 1
        else:
            b[i][0] = 1
    return b


train_data = np.loadtxt(open("./train.csv","rb"),delimiter=",",skiprows=0)
test_data = np.loadtxt(open("./predict.csv","rb"),delimiter=",",skiprows=0)
#selet rows and column
train_label_sex = train_data[:, 1:2]

train_label_sex = one_hot(train_label_sex,train_data.shape[0])


train_data = train_data[:, 3:]

train_data = np.reshape(train_data, (1858,n_steps,n_input))


test_label_sex = test_data[:, 1:2]
test_label_sex = one_hot(test_label_sex,test_data.shape[0])
test_data = test_data[:, 3:]
test_data = np.reshape(test_data, (200,n_steps,n_input))



x = tf.placeholder("float", [None, n_steps, n_input])

# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate = tf.placeholder("float", [None, 2 * n_hidden])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), 
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(_x, _istate, _weights, _biases):

    
    # Permuting  n_steps
  
    _x = tf.transpose(_x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    _x = tf.reshape(_x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)

	 
    _x = tf.matmul(_x, _weights['hidden']) + _biases['hidden']
    
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple=False)
    _x = tf.split(0, n_steps, _x) 
    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, _x, dtype=tf.float32, initial_state=_istate)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = RNN(x, istate, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    step = 1
    
    while step  < 300:
       
        sess.run(optimizer, feed_dict={x: train_data, y: train_label_sex, istate: np.zeros((training_iters, 2 * n_hidden))})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: train_data, y: train_label_sex, istate: np.zeros((training_iters, 2 * n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: train_data, y: train_label_sex,istate: np.zeros((training_iters, 2 * n_hidden))})
            print("Iter " + str(step) + ", Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")


    print("Testing Accuracy:", \
sess.run(accuracy, feed_dict={x: test_data, y: test_label_sex,
                                                             istate: np.zeros((Text_iters, 2 * n_hidden))}))
