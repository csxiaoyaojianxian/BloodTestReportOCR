# 血常规检验报告深度学习系统 on Spark

Spark是UC Berkeley AMP lab (加州大学伯克利分校的AMP实验室)所开源的类Hadoop MapReduce的通用并行框架，Spark，拥有Hadoop MapReduce所具有的优点；但不同于MapReduce的是Job中间输出结果可以保存在内存中，从而不再需要读写HDFS，因此Spark能更好地适用于数据挖掘与机器学习等需要迭代的MapReduce的算法。

该Demo主要演示Spark的深度学习功能，数据由Spark直接读取，尚未使用Hadoop等数据库。

##运行环境
###安装JDK
```
java -version
```
如果未安装，请下载最新JDK并设置相应的JAVA_HOME、JRE_HOME、CLASSPATH、PATH变量

###安装Scala并添加Scala_HOME,更新PATH

```
sudo apt-get install scala
```

下载Spark并解压

官网下载地址：http://spark.apache.org/downloads.html
###配置Spark环境
```
cp ./conf/spark-env.sh.template ./conf/spark-env.sh
```
###安装Python依赖包
```
sudo apt-get install python-numpy
```
###设置Python依赖路径
```
sudo vim /etc/profile
```
在结尾处添加
```
export SPARK_HOME=/home/hadoop/spark  #你的Spark解压目录

export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.1-src.zip:$PYTHONPATH   #py4j及pysqrk的相关依赖路径,py4j-0.10.1-src文件名可能会因Spark版本不同而不同，请设置为自己对应目录下的文件名
```
###启动SPARK
```
sudo ./sbin/start-all.sh
```
在root下输入jps应该可以看到Master和Worker两个进程

也可以登陆
```
http://127.0.0.1:8080/
```
查看Spark状态

##安装MongoDB Connector for Hadoop

MongoDB Connector for Hadoop是一个类库，他允许包括Spark、Pig、Hive、Mapreduce等在内的多种Hadoop架构中的组件使用MongoDB作为数据源。
###第三方软件安装
使用Maven安装：
```
<dependency>
    <groupId>org.mongodb.mongo-hadoop</groupId>
    <artifactId>mongo-hadoop-core</artifactId>
    <version>1.5.1</version>
</dependency>
```
或使用Gradle安装：
```
compile 'org.mongodb.mongo-hadoop:mongo-hadoop-core:1.5.1'
```
###独立安装
克隆源代码：
```
git clone https://github.com/mongodb/mongo-hadoop.git
```

源代码克隆后需要编译，编译过程需连接外网进行下载，国内下载速度较慢，建议使用VPN
```
./gradlew jar
```
编译后的文件会放在core/build/libs目录下。若安装了Hadoop，则将三个文件分别拷贝至以下目录

-$HADOOP_PREFIX/lib/
-$HADOOP_PREFIX/share/hadoop/mapreduce/
-$HADOOP_PREFIX/share/hadoop/lib/
若是Spark独立部署，则将其拷贝至本机pyspark目录下即可。

