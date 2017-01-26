# -*- coding: utf-8 -*-
from py_paddle import swig_paddle
import sys
sys.path.append("..")
from PaddlePaddle import prediction_sex,prediction_age
def predict(arr):
    swig_paddle.initPaddle("--use_gpu=0")
    data = [arr.tolist()]
    #直接填充4个0
    for i in range(4):
	data[0][0].append(0)
    sex = prediction_sex.predict(data)
    age = prediction_age.predict(data)
    return sex,age
