# -*- coding: UTF-8 -*-
import io
import random
import paddle.utils.image_util as image_util
from paddle.trainer.PyDataProvider2 import *
import csv

@provider(input_types=[
    #训练数据大小
    dense_vector(26),
    #标签种类
    integer_value(2)
])
#提供性别训练数据的函数
def process_sex(settings, file_name):
    csvfile = file('train.csv', 'rb')
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0]!='id':
	    sex=0
	    if(row[1]=='\xc4\xd0'):
		sex=1
	    del row[0]
	    del row[0]
	    del row[0]
	    pixels = []
	    for j in row:
		if(j!=''):
            	    pixels.append(float(j))
	    if(len(pixels)==26):
	    	yield pixels,int(sex)
    csvfile.close()

def predict_initializer(settings, **kwargs):
    settings.input_types=[
    dense_vector(26)
    ]
#提供性别预测数据的函数
@provider(init_hook=predict_initializer, should_shuffle=False)
def process_predict_sex(settings, file_name):
    csvfile = file('predict.csv', 'rb')
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    #预测第一行
    row=rows[1]
    sex='女'
    if(row[1]=='\xc4\xd0'):
	sex='男'
    print '实际性别：'+sex
    del row[0]
    del row[0]
    del row[0]
    pixels = []
    for j in row:
	pixels.append(float(j))
    if(len(pixels)==26):
	yield pixels

@provider(input_types=[
    dense_vector(26),
    integer_value(100)
])
#提供年龄训练数据的函数
def process_age(settings, file_name):
    csvfile = file('train.csv', 'rb')
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0]!='id':
	    age=int(row[2])
	    del row[0]
	    del row[0]
	    del row[0]
	    pixels = []
	    for j in row:
		if(j!=''):
            	    pixels.append(float(j))
	    if(len(pixels)==26):
	    	yield pixels,age
    csvfile.close()

def predict_initializer(settings, **kwargs):
    settings.input_types=[
    dense_vector(26)
    ]
#提供年龄预测数据的函数
@provider(init_hook=predict_initializer, should_shuffle=False)
def process_predict_age(settings, file_name):
    csvfile = file('predict.csv', 'rb')
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    row=rows[1]
    print '实际年龄：'+row[2]
    del row[0]
    del row[0]
    del row[0]
    pixels = []
    for j in row:
	if(j!=''):
            pixels.append(float(j))
    if(len(pixels)==26):
	yield pixels
    csvfile.close()

def hook(settings, img_size, mean_img_size, num_classes, color, meta, use_jpeg,
         is_train, **kwargs):
    settings.mean_img_size = mean_img_size
    settings.img_size = img_size
    settings.num_classes = num_classes
    settings.color = color
    settings.is_train = is_train

    if settings.color:
        settings.img_raw_size = settings.img_size * settings.img_size * 3
    else:
        settings.img_raw_size = settings.img_size * settings.img_size

    settings.meta_path = meta
    settings.use_jpeg = use_jpeg

    settings.img_mean = image_util.load_meta(settings.meta_path,
                                             settings.mean_img_size,
                                             settings.img_size,
                                             settings.color)

    settings.logger.info('Image size: %s', settings.img_size)
    settings.logger.info('Meta path: %s', settings.meta_path)
    '''
PaddlePaddle的数据包括四种主要类型，和三种序列模式。其中，四种数据类型是

dense_vector 表示稠密的浮点数向量。
sparse_binary_vector 表示稀疏的零一向量，即大部分值为0，有值的位置只能取1
sparse_float_vector 表示稀疏的向量，即大部分值为0，有值的部分可以是任何浮点数
integer 表示整数标签。
而三种序列模式为

SequenceType.NO_SEQUENCE 即不是一条序列
SequenceType.SEQUENCE 即是一条时间序列
SequenceType.SUB_SEQUENCE 即是一条时间序列，且序列的每一个元素还是一个时间序列。
'''
    settings.input_types = [
        dense_vector(settings.img_raw_size),  # image feature
        integer_value(settings.num_classes)]  # labels

    settings.logger.info('DataProvider Initialization finished')
'''
@provider 是一个Python的 Decorator ，他可以将某一个函数标记成一个PyDataProvider
PyDataProvider是PaddlePaddle使用Python提供数据的推荐接口。使用该接口用户可以只关注如何从文件中读取每一条数据，而不用关心数据如何传输给PaddlePaddle，数据如何存储等等。该数据接口使用多线程读取数据，并提供了简单的Cache功能
init_hook可以传入一个函数。这个函数在初始化的时候会被调用。这个函数的参数是:

第一个参数是 settings 对象。这个对象和process的第一个参数一致。具有的属性有
settings.input_types 设置输入类型。参考 input_types
settings.logger 一个logging对象
其他参数都使用key word argument传入。这些参数包括paddle定义的参数，和用户传入的参数。
Paddle定义的参数包括:
is_train bool参数，表示这个DataProvider是训练用的DataProvider或者测试用的 DataProvider
file_list 所有文件列表。
用户定义的参数使用args在训练配置中设置。

注意，PaddlePaddle保留添加参数的权力，所以init_hook尽量使用 **kwargs , 来接受不使用的 函数来保证兼容性。
详见http://www.paddlepaddle.org/doc_cn/ui/data_provider/pydataprovider2.html
'''
@provider(init_hook=hook)
def processData(settings, file_name):
    """
    加载数据
    迭代每一批的所有图像和标签
    file_name: 批文件名
    """
    #使用pickle类来进行python对象的序列化，而cPickle提供了一个更快速简单的接口，如python文档所说的：“cPickle -- A faster pickle”
    data = cPickle.load(io.open(file_name, 'rb'))
    #list() 方法用于将元组转换为列表，元组与列表的区别在于元组的元素值不能修改，元组是放在括号中，列表是放于方括号中。
    indexes = list(range(len(data['images'])))
    if settings.is_train:
        random.shuffle(indexes)
    for i in indexes:
        if settings.use_jpeg == 1:
            img = image_util.decode_jpeg(data['images'][i])
        else:
            img = data['images'][i]
	#如果不是训练, 裁剪图像中心区域.否则随机裁剪,
        img_feat = image_util.preprocess_img(img, settings.img_mean,
                                             settings.img_size, settings.is_train,
                                             settings.color)
        label = data['labels'][i]
	'''
	包含yield语句的函数会被特地编译成生成器。当函数被调用时，他们返回一个生成器对象
	不像一般函数生成值后退出，生成器函数生成值后会自动挂起并暂停他们的执行和状态，他的本地变量将保存状态信息，这些信息在函数恢复时将再度有效
	执行到 yield时，processData 函数就返回一个迭代值，下次迭代时，代码从 yield的下一条语句继续执行
	'''
        yield img_feat.tolist(), int(label)
