# -*- coding: UTF-8 -*-
from py_paddle import swig_paddle, DataProviderConverter
from paddle.trainer.PyDataProvider2 import dense_vector
from paddle.trainer.config_parser import parse_config
import numpy as np
import csv
import os

def predict(data):
    path=os.path.split(os.path.realpath(__file__))[0]
    conf = parse_config(path+"/trainer_config_age.py", "is_predict=1")
    print conf.data_config.load_data_args
    network = swig_paddle.GradientMachine.createFromConfigProto(conf.model_config)
    network.loadParameters(path+"/output_age/pass-00099")
    converter = DataProviderConverter([dense_vector(26)])
    inArg = converter(data)
    network.forwardTest(inArg)
    output = network.getLayerOutputs("__fc_layer_0__")
    #print output
    prob = output["__fc_layer_0__"][0]
    #print prob
    lab = np.argsort(-prob)
    #print lab
    return lab[0]

if __name__ == '__main__':
    swig_paddle.initPaddle("--use_gpu=0")
    csvfile = file('predict.csv', 'rb')
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    row=rows[1]
    print '实际年龄：'+row[2]
    del row[0]
    del row[0]
    del row[0]
    data = [[[]]]
    for j in row:
	data[0][0].append(float(j))
    csvfile.close()
    print '预测年龄:'+str(predict(data))
