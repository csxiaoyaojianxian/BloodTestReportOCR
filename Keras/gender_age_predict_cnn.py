#-*- coding: UTF-8 -*-

import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
import csv
import string

# 性别是2分类 年龄是10分类
age = 10
gender = 2

#修改这个地方可以选择预测性别还是年龄
#但年龄的准确率不忍直视
test_what = gender

#数据的分组边界
splitor=1400

# 准备数据
age_orign = []
data_orign = []
sex_orign = []
with open('train.csv','rb') as precsv:
	reader = csv.reader(precsv)
	for line in reader:
		# 忽略第一行
		if reader.line_num == 1:
			continue
		if(line[1] == '\xc4\xd0'):
			sex_orign.append(0) # 性别数据
		else:
			sex_orign.append(1) 
		age_orign.append(int(float(line[2])/10)) # 年龄(按照10岁为一个阶段分组)
		data_orign.append(line[4:]) # 血检数据
		
# 将数据分为训练集和测试集		
age_train = np.array(age_orign[:splitor])
data_train = np.array(data_orign[:splitor])
sex_train = np.array(sex_orign[:splitor])

age_predict = np.array(age_orign[splitor:])
data_predict = np.array(data_orign[splitor:])
sex_predict = np.array(sex_orign[splitor:])

# 数据的维度（数据含有多少项）
data_dim = data_train.shape[1]


if test_what == age:
	XT =  data_train.reshape(-1,data_dim,1,1)
	YT =  np_utils.to_categorical(age_train,nb_classes=age)
	XT2 =  data_predict.reshape(-1,data_dim,1,1)
	YT2 =  np_utils.to_categorical(age_predict,nb_classes=age)
	output_dim = age
	loss_str = 'categorical_crossentropy'
else:
	XT =  data_train.reshape(-1,data_dim,1,1)
	YT =  np_utils.to_categorical(sex_train,nb_classes=gender)
	XT2 =  data_predict.reshape(-1,data_dim,1,1)
	YT2 =  np_utils.to_categorical(sex_predict,nb_classes=gender)
	output_dim = gender
	loss_str = 'binary_crossentropy'

#
model = Sequential()

# 
model.add( Convolution2D( 
	nb_filter=data_dim*data_dim, 
	nb_row=5, 
	nb_col=5,
	border_mode='same',
	input_shape=(data_dim,1,1) 
	))
model.add(Activation('relu'))

# pooling

model.add( MaxPooling2D(
	pool_size=(2,2),
	strides=(2,2),
	border_mode='same'	 
	))
	
model.add( Convolution2D(64,5,5,border_mode='same'))
model.add( Flatten())
model.add( Dense(1024) )
#model.add( Activation('relu'))

model.add( Activation('relu'))
model.add(Dense(output_dim))
model.add( Activation('softmax'))
adam = Adam(lr=0.0001)
model.compile(
	loss=loss_str,
	optimizer=adam,
	metrics=['accuracy']
	)

model.fit(XT,YT,nb_epoch=100,batch_size=32)
	
print '===='	
loss,accuracy = model.evaluate(XT2,YT2)
print loss
print accuracy




