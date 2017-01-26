
#基于Spark的Ocr手写字符识别系统Demo

##构造训练测试数据
###下载数据集
```

wget http://labfile.oss.aliyuncs.com/courses/593/data.csv
```
该数据集是https://www.shiyanlou.com/courses/593/labs/1966/document 中为反向神经网络训练用的数据集
###格式化数据集
Spark深度学习常用的两种训练数据格式为Labeled point和LibSVM，在此，我们使用Labeled point作为训练数据格式。


labeled point 是一个局部向量，要么是密集型的要么是稀疏型的，用一个label/response进行关联。在Spark里，labeled points 被用来监督学习算法。我们使用一个double数来存储一个label，因此我们能够使用labeled points进行回归和分类。
在二进制分类里，一个label可以是 0（负数）或者 1（正数）。在多级分类中，labels可以是class的索引，从0开始：0,1,2,......

本Demo采用朴素贝叶斯作为训练、预测模型，特征值必须是非负数。

程序在运行过程中先读取并格式化./data.csv中的数据，然后和网页前端传来的训练数据一起格式化为labeled points格式
新生成的LabeledPoints数据保存在LabeledPointsdata.txt中。

需要预测时，先将LabeledPointsdata.txt中的数据读取为Spark 专用 RDD 形式，然后训练到model中

##运行


###创建服务器
```
 python -m SimpleHTTPServer 3000
```

###加载服务器
```
 python server.py

```
###访问
```
 localhost:3000
```