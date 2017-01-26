# -*- coding: cp936 -*-
#基于Spark的线性支持向量机血常规检验报告深度学习系统
#2016.12.14


from __future__ import print_function

import sys
import math
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.util import MLUtils


if __name__ == "__main__":

    sc = SparkContext(appName="BloodTestReportPythonSVMExample")

    # 读取数据.
    print('Begin Load Data File!')
    sexData = MLUtils.loadLabeledPoints(sc, "LabeledPointsdata_sex.txt")
    print('Data File has been Loaded!')

    accuracySex = []

    for i in range(0,100):
        # 将数据随机分割为9:1，分别作为训练数据（training）和预测数据（test）.
        sexTraining, sexTest = sexData.randomSplit([0.9, 0.1])

        # 训练线性支持向量机模型.
        #print('Begin SVM tranning!')
        sexModel = SVMWithSGD.train(sexTraining, iterations=100)
        #print('Trainning over!')
        # 对test数据进行预测，输出预测准确度.
        sexPredictionAndLabel = sexTest.map(lambda p: (sexModel.predict(p.features), p.label))
        accuracySex.append(1.0 * sexPredictionAndLabel.filter(lambda (x, v): x == v).count() / sexTest.count())

    #AVG：平均数  MSE：均方差
    SexRDD = sc.parallelize(accuracySex)
    SexPAAVG = SexRDD.reduce(lambda x,y:x+y)/SexRDD.count()
    SexPAMSE = math.sqrt(SexRDD.map(lambda x:(x - SexPAAVG)*(x - SexPAAVG)).reduce(lambda x,y:x+y)/SexRDD.count())

    print('Sex Prediction Accuracy AVG:{}'.format(SexPAAVG))
    print('Sex Prediction Accuracy MSE:{}'.format(SexPAMSE))

    output = open('SVMResult.txt', 'w')
    output.write('Sex Prediction Accuracy AVG is:' + str(SexPAAVG) + "\n")
    output.write('Sex Prediction Accuracy MSE is:' + str(SexPAMSE) + "\n")
    for i in accuracySex:
        output.write(str(i)+",")
    output.write("\n")
    output.close()
    
