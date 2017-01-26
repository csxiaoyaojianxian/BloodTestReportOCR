# -*- coding: utf-8 -*-  
from keras.preprocessing.image import ImageDataGenerator  
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.advanced_activations import PReLU  
from keras.layers.convolutional import Convolution2D, MaxPooling2D  
from keras.optimizers import SGD, Adadelta, Adagrad  
from keras.utils import np_utils, generic_utils  
from six.moves import range   
import random  
import os  
from PIL import Image  
import numpy as np
from keras import backend
backend.set_image_dim_ordering('th')  
  
#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，图像大小28*28  
#如果是将彩色图作为输入,则将1替换为3，并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]  
def load_data():  
    data = np.empty((42000,1,28,28),dtype="float32")  
    label = np.empty((42000,),dtype="uint8")  
  
    imgs = os.listdir("./mnist")  
    num = len(imgs)  
    for i in range(num):
        if '.jpg' in imgs[i]:
            img = Image.open("./mnist/"+imgs[i])  
            arr = np.asarray(img,dtype="float32")  
            data[i,:,:,:] = arr  
            label[i] = int(imgs[i].split('.')[0])  
    return data,label 

#加载数据  
data, label = load_data()  
#打乱数据  
index = [i for i in range(len(data))]  
random.shuffle(index)  
data = data[index]  
label = label[index]  
print(data.shape[0], ' samples')  
  
#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数  
label = np_utils.to_categorical(label, 10)  
  
#开始建立CNN模型  
#生成一个model,可以通过向 Sequential模型传递一个layer的list来构造该model 
model = Sequential()  
  
#第一个卷积层，4个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。  
#border_mode边界模式  可以是valid或者full，valid只适用于完整的图像补丁的过滤器
model.add(Convolution2D(4, 5, 5, border_mode='valid', input_shape=(1,28,28)))#input_shape在后面的层可以推导出来，不需要为每一个层都指定这个参数  
model.add(Activation('tanh'))  
#model.add(Dropout(0.5))#训练过程更新参数随机断开一定比例的神经元连接，避免过拟合，它们在正向传播过程中对于下游神经元的贡献效果暂时消失了，反向传播时该神经元也不会有任何权重的更新。  
  
#第二个卷积层，8个卷积核，每个卷积核大小3*3。                                                                  4表示输入的特征图个数，等于上一层的卷积核个数  
#激活函数用tanh  
#采用maxpooling，poolsize为(2,2)  
model.add(Convolution2D(8, 3, 3, border_mode='valid'))  
model.add(Activation('tanh'))  
model.add(MaxPooling2D(pool_size=(2, 2)))  
  
#第三个卷积层，16个卷积核，每个卷积核大小3*3  
#激活函数用tanh  
#采用maxpooling，poolsize为(2,2)最大化池操作，也就是下采样，这里是2维代表两个方向（竖直，水平）的 ，对输入进行size为(2,2)的下采样操作的话，结果就剩下了输入的每一维度的一半，即总的结果是原输入的四分之一。 
model.add(Convolution2D(16,  3, 3, border_mode='valid'))  
model.add(Activation('tanh'))  
model.add(MaxPooling2D(pool_size=(2, 2)))  
  
#全连接层，先将前一层输出的二维特征图flatten为一维的，常常用在卷积层到全链接层的过度。  
#Dense就是隐藏层。16就是上一层输出的特征图个数。4是根据每个卷积层计算出来的：(28-5+1)得到24,(24-3+1)/2得到11，(11-3+1)/2得到4  
#全连接有128个神经元节点,初始化方式为normal  
model.add(Flatten())  
model.add(Dense(input_dim=256, output_dim=128))#256=16*4*4  
model.add(Activation('tanh'))  
  
#Softmax分类，输出是10类别  
model.add(Dense(input_dim=128, output_dim=10))  
model.add(Activation('softmax'))  
  
#开始训练模型  
#使用SGD优化函数  ，lr学习速率， momentum参数更新动量，decay是学习速率的衰减系数(每个epoch衰减一次),Nesterov的值是False或者True，表示使不使用Nesterov momentum
#model.compile里的参数loss就是损失函数(目标函数)， optimizer是使用的优化器，metrics列表包含评估模型在训练和测试时的网络性能的指标 
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)  
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  
  
   
#调用fit方法，就是一个训练过程  
#输入的数据，标签，进行梯度下降的时候每个batch包含的样本数，训练的轮数，是否打乱，日志显示（0不输出日志信息，1输出进度条，2每轮训练输出一条记录），是否显示精确度，选择作为验证集的比例
model.fit(data, label, batch_size=100, nb_epoch=1,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.2)






















