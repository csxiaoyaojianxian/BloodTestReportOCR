
# PaddlePaddle图像分类demo





## 安装PaddlePaddle

```

# 下载安装包

wget https://github.com/PaddlePaddle/Paddle/releases/download/V0.8.0b1/paddle-cpu-0.8.0b1-Linux.deb


# 安装

gdebi paddle-cpu-0.8.0b1-Linux.deb

如果 gdebi 没有安装,则需要使用 sudo apt-get install gdebi, 来安装 gdebi 。

或者使用下面一条命令安装.

dpkg -i paddle-cpu-0.8.0b1-Linux.deb

apt-get install -f
在 dpkg -i 的时候如果报一些依赖未找到的错误是正常的， 在 apt-get install -f 里会继续安装 PaddlePaddle

官方教程http://www.paddlepaddle.org/doc_cn/build_and_install/install/ubuntu_install.html


```


## 下载MNIST数据集

下载地址https://pan.baidu.com/s/1kUNBkyz

在当前目录建立data文件夹，将MNIST.rar里的train和test文件夹解压到data文件夹下

注该数据集将原版MNIST二进制文件中的图片提取出来分别放入train和test文件夹，用户可以自行添加图片到train和test文件夹下，但要修改源码中关于图像大小的参数



## 训练MNIST


```

sh preprocess.sh # 调用preprocess.py 预处理

sh train.sh # 调用vgg.py训练，该脚本文件可设置训练模型存放路径和训练线程数等参数

python prediction.py # 预测，注意设置其中模型路径model_path

```



## 训练性别


训练前把train.csv，predict.csv拷贝到当前路径
```

sh train_sex.sh # 调用trainer_config_sex.py训练，注意设置num_passes训练次数，训练三十次错误率能降到30%左右

sh predict_sex.sh # 调用trainer_config_sex.py预测，注意设置模型路径model_path

```



## 训练年龄


训练前把train.csv，predict.csv拷贝到当前路径
```

sh train_age.sh # 调用trainer_config_age.py训练，注意设置num_passes训练次数，如果以5分段预测，训练100次错误率在85%左右，不分段错误率在95%左右

sh predict_age.sh # 调用trainer_config_age.py预测，注意设置模型路径model_path

```



## preprocess.py 


预处理模块，将data文件夹下的图片转换为PaddlePaddle格式

转换后的数据存放在data/batches文件夹下



## vgg.py


训练模块，使用VGG网络训练，该网络在ILSVRC2014的图像分类项目上获第二名

训练后的模型存放在vgg_model/pass-n文件夹下，n表示第几次训练，每训练一次会生成一个模型文件夹，理论上训练次数越多的模型效果越好

注使用CPU训练速度很慢，平均训练一次需要近半小时，目前PaddlePaddle使用CPU训练出来的模型和GPU训练出来的模型不一样，所以用CPU训练只能用CPU预测，用GPU训练只能用GPU预测，而且用GPU预测要安装GPU版的PaddlePaddle和CUDA，cudnn,并且需要NVIDIA显卡支持，所以这里用的是CPU版的



## prediction.py


预测模块，其中image参数为要识别的图像路径



## dataprovider.py


实现向PaddlePaddle提供数据的接口，详见dataprovider.py注释



## trainer_config_sex.py


性别训练网络配置



## trainer_config_age.py


年龄训练网络配置



## predict_age.sh & predict_sex.sh


预测脚本文件，预测的结果保存在当前路径下的result.txt文件，第一个数为预测的结果，后面的数是每个标签的概率



## prediction_age.py & prediction_sex.py


提供预测接口，也可单独执行，接口输入为一个形如[[[0,1,2,...]]]的list，输出为性别或年龄的标签



## train.list & test.list


训练文件和测试文件的列表



## __init__.py


使A2的文件能导入本文件夹下的模块