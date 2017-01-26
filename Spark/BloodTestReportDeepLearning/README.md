# 基于Spark的血常规检验报告深度学习系统
##构造训练测试数据
原始数据在data_set.csv中，运行
```
python ./dataformat.py
```

生成Spark使用的labeled point数据，分别保存在LabeledPointsdata_age.txt和LabeledPointsdata_sex.txt中

##运行

所有示例都自动对两个数据集中的数据随机分为9：1，9份做模型训练，1份做预测测试。重复100次后分别计算年龄和性别的预测准确度和方差，在屏幕输出的同时，保存在对应的 算法名+result.txt文件中。

###朴素贝叶斯算法（支持多分类）
```
python ./BloodTestReportbyNB.py
```

结果：
```
Sex Prediction Accuracy AVG is:0.621970740283
Sex Prediction Accuracy MSE is:0.0339853457575
Age Prediction Accuracy AVG is:0.539635804425
Age Prediction Accuracy MSE is:0.039652048965
```
###线性支持向量机（仅支持二分类）
```
python ./BloodTestReportbySVM.py
```

结果(迭代次数=100)：
```
Sex Prediction Accuracy AVG is:0.528946440893
Sex Prediction Accuracy MSE is:0.0499342692342
```

###逻辑回归（仅支持二分类）

```
python ./BloodTestReportbyLR.py
```

结果(迭代次数=100)：
```
Sex Prediction Accuracy AVG is:0.717975697167
Sex Prediction Accuracy MSE is:0.0303414723843
```

###随机树（支持多分类）
```
python ./BloodTestReportbyRF.py
```

结果（树=3，最大深度=4，最大叶子数=32，纯度计算方式：基尼系数，性别分类=2，年龄分类=1000（此处取值与纯度计算方式有关，实际年龄label只有92个，具体算法还未完全掌握））：
```
Sex Prediction Accuracy AVG is:0.71622711581
Sex Prediction Accuracy MSE is:0.0255871783772
Age Prediction Accuracy AVG is:0.561977173151
Age Prediction Accuracy MSE is:0.0622593472121
```

###梯度提升树（仅支持二分类）

```
python ./BTR_gradient_boosting.py
```

结果(迭代次数=100)：
```
Sex Prediction Accuracy AVG is:0.728212518228
Sex Prediction Accuracy MSE is:0.0305777571064
```

###二分类（仅支持二分类）

```
python ./BTR_binary_classification.py
```

结果(迭代次数=100)：
```
Sex Prediction Accuracy AVG is:0.718756411999
Sex Prediction Accuracy MSE is:0.0311279215968
```

###决策树（支持多分类）
```
python ./BTR_decision_tree.py
```

结果：
```
Sex Prediction Accuracy AVG is:0.707608775434
Sex Prediction Accuracy MSE is:0.0292234440441
Age Prediction Accuracy AVG is:0.552560046229
Age Prediction Accuracy MSE is:0.05098502703
```