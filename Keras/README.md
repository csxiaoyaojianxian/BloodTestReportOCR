Keras手写识别字符集Demo

1.关于Keras

  Keras是基于Theano和TensorFlow的一个深度学习框架，它的设计参考了Torch，用Python语言编写，是一个高度模块化的神经网络库，支持GPU和CPU。
   
2.关于Keras的Sequential模型

  Keras有两种类型的模型，顺序模型（Sequential）和泛型模型（Model）：
  
  2.1 Sequential是多个网络层的线性堆叠，可以通过向Sequential模型传递一个layer的list来构造该模型：
		from keras.models import Sequential
		from keras.layers import Dense, Activation
		model = Sequential([Dense(32, input_dim=784),Activation('relu'),Dense(10),Activation('softmax'),])
	  也可以通过.add()方法一个个的将layer加入模型中：
		model = Sequential()
		model.add(Dense(32, input_dim=784))
		model.add(Activation('relu'))
		
  2.2 模型需要知道输入数据的shape
	  Sequential的第一层需要接受一个关于输入数据shape的参数，后面的各个层则可以自动的推导出中间数据的shape，因此不需要为每个层都指定这个参数。
      
	  有几种方法来为第一层指定输入数据的shape：
		1.传递一个input_shape的关键字参数给第一层，input_shape是一个tuple类型的数据，其中也可以填入None，如果填入None则表示此位置可能是任何正整数。数据的batch大小不应包含在其中。
		2.传递一个batch_input_shape的关键字参数给第一层，该参数包含数据的batch大小。该参数在指定固定大小batch时比较有用，例如在stateful RNNs中。事实上，Keras在内部会通过添加一个None将input_shape转化为batch_input_shape
		3.有些2D层，如Dense，支持通过指定其输入维度input_dim来隐含的指定输入数据shape。一些3D的时域层支持通过参数input_dim和input_length来指定输入shape。
	  下面的三个指定输入数据shape的方法是严格等价的：
		1.model = Sequential()
		  model.add(Dense(32, input_shape=(784,)))
		2.model = Sequential()
		  model.add(Dense(32, batch_input_shape=(None, 784)))
		3.model = Sequential()
		  model.add(Dense(32, input_dim=784))
		  
  2.3 Sequential模型常用方法
  
		2.3.1 compile
		compile(self, optimizer, loss, metrics=[], sample_weight_mode=None)
		编译用来配置模型的学习过程，其参数有：
			optimizer：字符串（预定义优化器名）或优化器对象，参考优化器
			loss：字符串（预定义损失函数名）或目标函数，参考目标函数
			metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是metrics=['accuracy']
			sample_weight_mode：如果你需要按时间步为样本赋权（2D权矩阵），将该值设为“temporal”。默认为“None”，代表按样本赋权（1D权）。在下面fit函数的解释中有相关的参考内容。
			kwargs：使用TensorFlow作为后端请忽略该参数，若使用Theano作为后端，kwargs的值将会传递给 K.function
		代码示例：
			model = Sequential()
			model.add(Dense(32, input_shape=(500,)))
			model.add(Dense(10, activation='softmax'))
			model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
			
		2.3.2 fit
		fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
		本函数将模型训练nb_epoch轮，其参数有：
			x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
			y：标签，numpy array
			batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
			nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为"number of"的意思
			verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
			callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
			validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。
			validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
			shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。
			class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）
			sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。

3.关于Keras的常用层

	3.1 Dense层
		Dense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
		Dense就是常用的全连接层，这里是一个使用示例：
			model = Sequential()
			model.add(Dense(32, input_dim=16))

			model = Sequential()
			model.add(Dense(32, input_shape=(16,)))
			
			model.add(Dense(32))
		部分常用参数：
			output_dim：大于0的整数，代表该层的输出维度。模型中非首层的全连接层其输入维度可以自动推断，因此非首层的全连接定义时不需要指定输入维度。
			init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递weights参数时才有意义。
			activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
			input_dim：整数，输入数据的维度。当Dense层作为网络的第一层时，必须指定该参数或input_shape参数。
			
	3.2 Activation层
		Activation(activation)
		激活层对一个层的输出施加激活函数。
		
	3.3 Dropout层
		Dropout(p)
		为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接，Dropout层用于防止过拟合。
		
	3.4 Flatten层
		Flatten()
		Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。这里是一个使用例子：
			model = Sequential()
			model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
			# 模型输出形状 == (None, 64, 32, 32)
			model.add(Flatten())
			# 模型输出形状 == (None, 65536)
			
	3.5 Convolution1D层
		Convolution1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
		一维卷积层，用以在一维输入信号上进行邻域滤波。当使用该层作为首层时，需要提供关键字参数input_dim或input_shape。
		
	3.6 Convolution2D层
		二维卷积层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数。
		
	3.7 MaxPooling1D层
		MaxPooling1D(pool_length=2, stride=None, border_mode='valid')
		对时域1D信号进行最大值池化
		参数：
			pool_length：下采样因子，如取2则将输入下采样到一半长度
			stride：整数或None，步长值
			border_mode：‘valid’或者‘same’
 
 4.数据集 
 
	kerashandwerite.py的数据集下载地址：http://pan.baidu.com/s/1nvEuc8D
	
## 基于Keras的CNN性别年龄预测预测
本次request通过keras实现了性别预测

- 性别预测：cnn方法将train.csv的数据二分，对训练集本身的准确率在一段时间后会极高，但预测集在73%-78%。进一步增加训练次数后，反而将导致测试准确率下降。

- 但是年龄预测由于训练数据过少，分类较多，目前无法得到明显的准确率提升，仅能维持在10选1的概率下达到26%-30%。


### 其他简要说明

1. gender-predict-cnn实现了一个初级的cnn性别预测算法。

2. 下一步方向是参数调优和尝试使用Inception-v4或者ResNet/VGG来实现性别或年龄预测(这可以在数据不全的场合下提高准确率)。

3. （年龄预测可以尝试按照gender-predict-cnn中的注释修改实现预测）

4. 数据集为train.csv

5. 具体函数作用参看注释和本节之前的说明。

