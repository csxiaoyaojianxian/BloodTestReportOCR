# Tensorflow框架下的mnist手写字符识别
- 简单双隐层 26->238->512->100（年龄）/ 2（性别）
- 学习率0.01/0.1
- 训练数据集A2的csv血液数据报告文件
- 输出层用softmax函数做分类器，损失函数是cross entropy
- 批处理大小为17
- 本内容皆在提供tensorflow标准数据格式的预处理范例

### 环境配置
系统: UBUNTU系列， 有N卡支持CUDA请装GPU版本并在sess出处使用GPU执行训练

    # 安装numpy
    sudo apt-get install python-numpy
    
    # 安装PIL
    sudo apt-get install python-imaging
    
    # 安装Tensorflow
    pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl
    
    
### 运行
    mkdir ckpt_age
    mkdir ckpt_sex
    python age_predict.py   # 网络结构未优化，准确率40%上下
    python sex_predict.py   # 同样由于网络结构问题，损失函数不收敛
### 解释
1. age_predict.py 
   训练网络，并预测一条记录（预测样本放在代码最后）
2. tfrecords后缀文件和ckpt文件夹下内容 
    第一次运行会根据数据集产生tfrecord文件，文件feed以及分batch均需要构建为这个标准数据格式，如需要扩充变化数据集请删除tfrecords内容（如要变化数据格式请重新）
    第一次运行会在ckpt下状态保存点，如果需要调参再训练，请删除ckpt文件夹下内容; 

### 注意
如果不是用的最新版tensorflow，请去旧版文档查询并更改Saver()和Initializer()函数，0.11及以下版本使用的API名称是不同的

### agepredictv2.0.py注释
定义了添加层函数。通过升维，使不同年龄段输出节点不同，调参找到比较好结果，设置2隐藏层。隐藏层节点数约为输入75%。使年龄预测率提高到24%左右。
其典型的bp神经网络模型流要更具有普适性。可直接利用该文件夹俩csv文件运行。 ——SA312

### TensorBoard可视化
程序运行完毕之后, 会产生logs目录 , 使用命令 tensorboard --logdir='logs/'，然后打开浏览器查看
