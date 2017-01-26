##文件说明

 - caffe_sex_train_predict.py 性别预测demo主要代码，完成数据格式转换，训练及预测流程控制
 - config.prototxt                     训练网络配置文件
 - lenet_train.prototxt              训练网络设置
 - model_prod_prototxt           预测网络设置
 - draw_net.py                         网络绘图代码（未整合至主代码文件中）

##caffe的安装：
**1、安装基本依赖**

```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
```

```
sudo apt-get install --no-install-recommends libboost-all-dev
```

	由于ubuntu的库有各种依赖关系，apt－get可能无法解决，建议使用aptitude，会给出多个解决方案，实测可行！
	sudo aptitude install ...

**2、若不使用gpu，可以跳过安装cuda！（而且好像16.04已经带有cuda8）**

**3、安装ATLAS**

```
sudo apt-get install libatlas-base-dev
```

**4、下载caffe**

```
git clone https://github.com/BVLC/caffe.git
```

**5、修改Makefile.config**

```
cd caffe
cp Makefile.config.example Makefile.config
gedit Makefile.config
```

将# cpu_only := 1的注释去掉，找到并修改为：

```
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/i386-linux-gnu/hdf5/serial
```
如果是ubuntu16.04 64位版本，需要将第二项改为 ：
```
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial
```

如果make all依然有错，你可能需要进行下一步
```
cd /usr/lib/x86_64-linux-gnu

sudo ln -s libhdf5_serial.so.10.1.0 libhdf5.so

sudo ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so
```
这依然是版本的锅。

**6、编译安装**

```
make all
make test
make runtest
```

到此caffe安装已经完成！
若有需要用到python或matlab接口的，先设置好Makefile.config中的路径，再另外编译：

```
make pycaffe
make matcaffe
```
ubuntu16.04 64位出错可能的解决方法：
```
# (Python 2.7 development files)
sudo apt-get install -y python-dev
sudo apt-get install -y python-numpy python-scipy
```
修改Makefile.config中
```
PYTHON_INCLUDE := /usr/include/python2.7 /usr/local/lib/python2.7/dist-packages/numpy/core/include

WITH_PYTHON_LAYER := 1
```
这是因为numpy安装路径可能不一样。

添加python环境变量，方便以后imoprt caffe，打开/etc/bash.bashrc末尾添加：

```
PYTHONPATH=/xxx/xxx/caffe/python:$PYTHONPATH
```

另外pycaffe的接口暴露在caff目录下的python文件夹，只需要import caffe就可以直接调用。matcaffe接口官网有介绍。

##prototxt网络模型绘制成可视化图片

draw_net.py可以将网络模型由prototxt变成一张图片，draw_net.py存放在caffe根目录下python文件夹中。

绘制网络模型前，先安装两个库：ＧraphViz和pydot

**1.安装ＧraphViz**

Graphviz的是一款图形绘制工具，用来被python程序调用绘制图片

    sudo apt-get install GraphViz

**2.安装pydot**

pydot是python的支持画图的库

    sudo pip install pydot

**3.编译pycaffe**

    make pycaffe

完成上面三个步骤之后，就可以绘制网络模型了，draw_net.py执行的时候带三个参数

第一个参数：网络模型的prototxt文件

第二个参数：保存的图片路径及名字

第二个参数：–rankdir=x , x 有四种选项，分别是LR, RL, TB, BT 。用来表示网络的方向，分别是从左到右，从右到左，从上到小，从下到上。默认为ＬＲ。

**绘制Lenet模型**

在caffe根目录下

    python python/draw_net.py examples/mnist/lenet_train_test.prototxt ./lenet_train_test.jpg --rankdir=BT

绘制完成后将会生成lenet_train_test.jpg

## 利用CAFFE预测病人性别,正确率只有70%，还可以通过优化网络结构进行提升

### 环境配置(Ubuntu 14.04或以上版本)

如果还有模块没有安装，可以使用如下命令安装
```
sudo pip install module_name
```
获取的数据来源：

同项目目录下`Spark/BllodTestReportDeeplearning/data_set.csv`

### 使用
 - 在当前目录下建立两个数据库文件夹，test_data_lmdb，train_data_lmdb

```
mkdir test_data_lmdb train_datalmdb
```
 - 运行caffe_sex_train_predict.py

```
python caffe_sex_train_predict.py
```

注意：重复运行create_data_lmdb()并不会覆盖原来的文件，而是会在原文件结尾处继续生成新数据，如
果需要重新调试，可以删除两个文件

相关资料链接：
官网上神经网络搭建实例：
http://nbviewer.ipython.org/github/joyofdata/joyofdata-articles/blob/master/deeplearning-with-caffe/Neural-Networks-with-Caffe-on-the-GPU.ipynb

layer 详解：
http://blog.csdn.net/u011762313/article/details/47361571#sigmoid
