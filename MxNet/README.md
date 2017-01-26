*Mxnet是一个轻量化分布式可移植深度学习计算平台，支持多机多节点、多GPU的计算，其openMP+MPI/SSH+Cuda/Cudnn的框架是的计算速度很快，且能与分布式文件系统结合实现大数据的深度学习。*
##Mxnet单节点的安装：
**1、安装基本依赖**
```
sudo apt-get update
```

```
sudo apt-get install -y build-essential git libblas-dev libopencv-dev
```
**2、下载mxnet**
```
git clone --recursive https://github.com/dmlc/mxnet.git
```
**3、安装CUDA**
```
具体参见http://blog.csdn.net/xizero00/article/details/43227019
```
**4、编译支持GPU的MXnet**

将mxnet/目录里找到mxnet/make/子目录，把该目录下的config.mk复制到mxnet/目录，用文本编辑器打开，找到并修改以下两行：
```
USE_CUDA = 1

USE_CUDA_PATH = /usr/local/cuda
```
修改之后，在mxnet/目录下编译
```
make -j4
```
**５、安装Python支持**
```
cd python;

python setup.py install
```
有些时候需要安装setuptools和numpy(sudo apt-get install python-numpy)。
**６、运行Mnist手写体识别实例**
在mxnet/example/image-classification里可以找到MXnet自带MNIST的识别样例
```
cd mxnet/example/image-classification

python train_mnist.py
```