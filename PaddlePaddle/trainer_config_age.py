# -*- coding: UTF-8 -*-
from paddle.trainer_config_helpers import *
import csv

is_predict = get_config_arg('is_predict', bool, False)
define_py_data_sources2(
    #训练文件列表
    train_list='train.list' if not is_predict else None,
    #测试文件列表
    test_list='test.list',
    #指明提供数据的函数
    module="dataprovider",
    obj='process_age' if not is_predict else 'process_predict_age')

settings(
    #批尺寸
    batch_size=128 if not is_predict else 1,
    #学习速率
    learning_rate=2e-3,
    #学习方式
    learning_method=AdamOptimizer(),
    #权重衰减
    regularization=L2Regularization(8e-4))
#输入数据大小
data = data_layer(name="data", size=26)
#直接全连接，指明输出数据大小，激活函数是Softmax
output = fc_layer(name="__fc_layer_0__",input=data, size=100, act=SoftmaxActivation())
if is_predict:
    #获得最大概率的标签
    maxid = maxid_layer(output)
    outputs([maxid, output])
else:
    #标签大小
    label = data_layer(name="label", size=100)
    #计算误差
    cls = classification_cost(input=output, label=label)
    outputs(cls)
