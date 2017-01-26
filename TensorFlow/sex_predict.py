# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import csv
import random
# id,sex,age,WBC,RBC,HGB,HCT,MCV,MCH,MCHC,RDW,PLT,MPV,PCT,PDW,LYM,LYM%,MON,MON%,NEU,NEU%,EOS,EOS%,BAS,BAS%,ALY,ALY%,LIC,LIC%

# 1为男, 0为女

'''
数据处理部分
'''
# 数据集路径
cwd = os.getcwd()

train = csv.reader(open(cwd + '/train.csv', 'rb'))
predict = csv.reader(open(cwd + '/predict.csv', 'rb'))


# 转化标签为one-hot格式(类别为两类，男和女)
def dense_to_one_hot(labels_dense, num_classes=2):
    labels_dense = np.array(labels_dense)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# 读取数据
def write_to_tensor(name, csv_name):
    if os.path.exists(name):
        return
    csv_file = csv.reader(open(cwd + '/' + csv_name, 'rb'))
    writer = tf.python_io.TFRecordWriter(name)
    for line in csv_file:
        if not line:
            break
        if len(line) is not 29:
            continue
        index = [int(line[1])]
        # 提取从第4列到第28列
        data = map(float, line)[3:29]
        # 注意list类型, Feature或FeatureList等
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=index)),
            'content': tf.train.Feature(float_list=tf.train.FloatList(value=data))
        }))
        print data, index
        # 序列化并写入tfrecord
        writer.write(example.SerializeToString())
    writer.close()


# 读取数据并解析
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    # 创建tfrecord reader
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    # 读取时要注意fix shape
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'content': tf.FixedLenFeature([26], tf.float32),
                                       })
    data = tf.cast(features['content'], tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return data, label


'''
网络结构部分  结构 双隐层  26 -  64 - 512 - 2 均为全连接层
'''

#添加层函数
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Ws = tf.Variable(tf.random_normal([in_size,out_size]))
            tf.histogram_summary(layer_name+'/weights',Ws)
        with tf.name_scope('baises'):
            bs = tf.Variable(tf.zeros([1,out_size])+0.5)
            tf.histogram_summary(layer_name+'/baises',bs)
        with tf.name_scope('Wx_plus_b'):
            Wxpb = tf.matmul(inputs,Ws) + bs
  
        if activation_function is None:
            outputs = Wxpb
        else:
            outputs = activation_function(Wxpb)
        tf.histogram_summary(layer_name+'/outputs',outputs)
        return outputs

# 定义占位符
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, shape=[None, 26])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])

#2个隐藏层
l1 = add_layer(tf.reshape(x, [-1, 26]),26,64,n_layer=1,activation_function=tf.nn.relu)
l2 = add_layer(l1,64,512,n_layer=2,activation_function=tf.nn.relu)
# add output layer
y_result = add_layer(l2,512,2,n_layer=3)


# 定义损失函数 交叉熵
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_result, y_))
    tf.scalar_summary('loss',cross_entropy)
# 定义训练op
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

# 定义正确预测
# correct_prediction = tf.less_equal(tf.abs(tf.sub(tf.argmax(y_result, 1), tf.argmax(y_, 1))), 5)
correct_prediction = tf.equal(tf.argmax(y_result, 1), tf.argmax(y_, 1))

# 定义正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 定义Model Saver op
saver = tf.train.Saver()

# 定义计算图激活op
init_op = tf.global_variables_initializer()

'''
训练部分
'''
# 如果没有保存模型则训练一个新的
if not os.path.exists("./ckpt_sex/checkpoint"):
    # 创建tfrecord
    write_to_tensor('train_sex.tfrecords', 'train.csv')
    write_to_tensor('predict_sex.tfrecords', 'predict.csv')
    # 读取tfrecord
    train_img, train_label = read_and_decode("train_sex.tfrecords")
    test_img, test_label = read_and_decode("predict_sex.tfrecords")

    # 使用shuffle_batch分batch并打乱顺序
    img_batch, label_batch = tf.train.shuffle_batch([train_img, train_label],
                                                        batch_size=17, capacity=2000,
                                                        min_after_dequeue=1000)
    test_img_batch, test_label_batch = tf.train.shuffle_batch([test_img, test_label],
                                                                  batch_size=200, capacity=20000,
                                                                  min_after_dequeue=10000)
    with tf.Session() as sess:
        
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("logs/", sess.graph)
        # 激活计算图
        sess.run(init_op)
        # 启动队列
        threads = tf.train.start_queue_runners(sess=sess)
        # 迭代次数 = 10000
        for i in range(10000):
            # batch
            image, label = sess.run([img_batch, label_batch])
            # 输出局部正确率
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: image, y_: dense_to_one_hot(label)})
                print("step %d, training accuracy %g" % (i, train_accuracy))
                result = sess.run(merged,feed_dict={x: image, 
                                                    y_: dense_to_one_hot(label)})
                writer.add_summary(result,i)
            train_step.run(feed_dict={x: image, y_: dense_to_one_hot(label)})
        # 加载测试集
        test_img, test_label = sess.run([test_img_batch, test_label_batch])
        # 输出整体正确率
        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: test_img, y_: dense_to_one_hot(test_label)}))
        # 保存模型
        save_path = saver.save(sess, cwd + "/ckpt_sex/sex.ckpt", write_meta_graph=None)
        print("Model saved in file: %s" % save_path)

'''
预测部分
给出预测数据格式CSV中任意一行(包括id有29个数据)即可
'''

def preloadedata(data):
    return tf.reshape(np.array(map(float, data[3:29])), [1, 26]).eval()

# 加载模型
with tf.Session() as sess:
    # 恢复checkpoint.
    saver.restore(sess, cwd + "/ckpt_sex/sex.ckpt")
    print("Model restored.")
    # 读取数据
    predict_data = csv.reader(open(cwd + '/predict.csv', 'rb'))
    # 预处理数据
    my_data = [37,1,66,8.7,6.9,0.111,10.8,0.55,6.3,0.4,4.6,7.61,87.7,0.1,3.78,1.1,0.03,0.3,0.03,0.4,0.16,1.8,122,0.352,93,32.1,345,11.4,160]
    my_data = preloadedata(my_data)
    # 输出预测结果
    print "predictions", tf.argmax(y_result, 1).eval(feed_dict={x: my_data, keep_prob: 1.0}, session=sess)
    # 输出男女概率
    print "probabilities", tf.nn.softmax(y_result.eval(feed_dict={x: my_data, keep_prob: 1.0}, session=sess)).eval()

