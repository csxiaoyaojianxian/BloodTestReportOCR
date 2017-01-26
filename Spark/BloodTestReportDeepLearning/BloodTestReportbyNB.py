# -*- coding: cp936 -*-
#基于Spark的朴素贝叶斯血常规检验报告深度学习系统
#2016.12.14

from __future__ import print_function

import sys
import math
from pyspark import SparkContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils


if __name__ == "__main__":

    sc = SparkContext(appName="BloodTestReportPythonNaiveBayesExample")

    # 读取数据.
    print('Begin Load Data File!')
    sexData = MLUtils.loadLabeledPoints(sc, "LabeledPointsdata_sex.txt")
    ageData = MLUtils.loadLabeledPoints(sc, "LabeledPointsdata_age.txt")
    #print(data.collect())
    print('Data File has been Loaded!')
    #for(d in data.take(3)):
    #   print(d)
    accuracySex = []
    accuracyAge = []
    for i in range(0,100):
        # 将数据随机分割为9:1，分别作为训练数据（training）和预测数据（test）.
        sexTraining, sexTest = sexData.randomSplit([0.9, 0.1])
        ageTraining, ageTest = ageData.randomSplit([0.9, 0.1])

        # 训练朴素贝叶斯模型.
        #print('Begin NaiveBayes tranning!')
        sexModel = NaiveBayes.train(sexTraining, 1.0)
        ageModel = NaiveBayes.train(ageTraining, 1.0)
        #print('Trainning over!')
        # 对test数据进行预测，输出预测准确度.
        sexPredictionAndLabel = sexTest.map(lambda p: (sexModel.predict(p.features), p.label))
        agePredictionAndLabel = ageTest.map(lambda p: (ageModel.predict(p.features), p.label))
        #print(predictionAndLabel.collect())
        accuracySex.append(1.0 * sexPredictionAndLabel.filter(lambda (x, v): x == v).count() / sexTest.count())
        accuracyAge.append(1.0 * agePredictionAndLabel.filter(lambda (x, v): abs((x-v)<=5)).count() / ageTest.count())
    #AVG：平均数  MSE：均方差
    SexRDD = sc.parallelize(accuracySex)
    AgeRDD = sc.parallelize(accuracyAge)
    SexPAAVG = SexRDD.reduce(lambda x,y:x+y)/SexRDD.count()
    AgePAAVG = AgeRDD.reduce(lambda x,y:x+y)/AgeRDD.count()
    SexPAMSE = math.sqrt(SexRDD.map(lambda x:(x - SexPAAVG)*(x - SexPAAVG)).reduce(lambda x,y:x+y)/SexRDD.count())
    AgePAMSE = math.sqrt(AgeRDD.map(lambda x:(x - AgePAAVG)*(x - AgePAAVG)).reduce(lambda x,y:x+y)/AgeRDD.count())
    #print(sum(accuracySex) / len(accuracySex))
    #print(sum(accuracyAge) / len(accuracyAge))

    print('Sex Prediction Accuracy AVG:{}'.format(SexPAAVG))
    print('Sex Prediction Accuracy MSE:{}'.format(SexPAMSE))
    print('AGE Prediction Accuracy AVG:{}'.format(AgePAAVG))
    print('AGE Prediction Accuracy MSE:{}'.format(AgePAMSE))

    output = open('NaiveBayesResult.txt', 'w')
    output.write('Sex Prediction Accuracy AVG is:' + str(SexPAAVG) + "\n")
    output.write('Sex Prediction Accuracy MSE is:' + str(SexPAMSE) + "\n")
    for i in accuracySex:
        output.write(str(i)+",")
    output.write("\n")
    output.write('Age Prediction Accuracy AVG is:' + str(AgePAAVG) + "\n")
    output.write('Age Prediction Accuracy MSE is:' + str(AgePAMSE) + "\n")
    for i in accuracyAge:
        output.write(str(i) + ",")
    output.write("\n")
    output.close()
    
