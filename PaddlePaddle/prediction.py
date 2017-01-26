# -*- coding: UTF-8 -*-
import os,sys
import numpy as np
import logging
from PIL import Image
from optparse import OptionParser

import paddle.utils.image_util as image_util

from py_paddle import swig_paddle, DataProviderConverter
from paddle.trainer.PyDataProvider2 import dense_vector
from paddle.trainer.config_parser import parse_config

logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')
logging.getLogger().setLevel(logging.INFO)

class ImageClassifier():
    def __init__(self,
                 train_conf,
                 use_gpu=True,
                 model_dir=None,
                 resize_dim=None,
                 crop_dim=None,
                 mean_file=None,
                 oversample=False,
                 is_color=False):
        """
        train_conf: 网络配置文件
        model_dir: 模型路径
        resize_dim: 设为原图大小
        crop_dim: 图像裁剪大小，一般设为原图大小
        oversample: bool, oversample表示多次裁剪,这里禁用
        """
        self.train_conf = train_conf
        self.model_dir = model_dir
        if model_dir is None:
            self.model_dir = os.path.dirname(train_conf)

        self.resize_dim = resize_dim
        self.crop_dims = [crop_dim, crop_dim]
        self.oversample = oversample
        self.is_color = is_color

        self.transformer = image_util.ImageTransformer(is_color = is_color)
        self.transformer.set_transpose((2,0,1))

        self.mean_file = mean_file
        mean = np.load(self.mean_file)['data_mean']
        mean = mean.reshape(1, self.crop_dims[0], self.crop_dims[1])
        self.transformer.set_mean(mean) # mean pixel
        gpu = 1 if use_gpu else 0
        conf_args = "is_test=1,use_gpu=%d,is_predict=1" % (gpu)
	#使用 parse_config() 解析训练时的配置文件
        conf = parse_config(train_conf, conf_args)
	#PaddlePaddle目前使用Swig对其常用的预测接口进行了封装，使在Python环境下的预测接口更加简单
	#使用 swig_paddle.initPaddle() 传入命令行参数初始化 PaddlePaddle
        swig_paddle.initPaddle("--use_gpu=%d" % (int(use_gpu)))
	#使用 swig_paddle.GradientMachine.createFromConfigproto() 根据上一步解析好的配置创建神经网络
        self.network = swig_paddle.GradientMachine.createFromConfigProto(conf.model_config)
        assert isinstance(self.network, swig_paddle.GradientMachine)
	#从模型文件加载参数
        self.network.loadParameters(self.model_dir)

        data_size = 1 * self.crop_dims[0] * self.crop_dims[1]
        slots = [dense_vector(data_size)]
	'''
创建一个 DataProviderConverter 对象converter。
swig_paddle接受的原始数据是C++的Matrix，也就是直接写内存的float数组。 这个接口并不用户友好。所以，我们提供了一个工具类DataProviderConverter。 这个工具类接收和PyDataProvider2一样的输入数据
	'''
        self.converter = DataProviderConverter(slots)

    def get_data(self, img_path):
        """
        1. 读取图片.
        2. resize 或 oversampling.
        3. transformer data: transpose, sub mean.
        return K x H x W ndarray.
        """
        image = image_util.load_image(img_path, self.is_color)
        if self.oversample:
            # image_util.resize_image: short side is self.resize_dim
            image = image_util.resize_image(image, self.resize_dim)
            image = np.array(image)
            input = np.zeros((1, image.shape[0], image.shape[1],1),
                             dtype=np.float32)
	    if self.is_color:
            	input[0] = image.astype(np.float32)
	    else:
	    	for i in range(0,self.resize_dim):
		    for j in range(0,self.resize_dim):
		        input[0][i][j][0]=image[i][j]
            input = image_util.oversample(input, self.crop_dims)
        else:
            image = image.resize(self.crop_dims, Image.ANTIALIAS)
	    image = np.array(image)
            input = np.zeros((1, self.crop_dims[0], self.crop_dims[1],1),
                             dtype=np.float32)
	    if self.is_color:
            	input[0] = image.astype(np.float32)
	    else:
	    	for i in range(0,self.resize_dim):
		    for j in range(0,self.resize_dim):
		        input[0][i][j][0]=image[i][j]

        data_in = []
        for img in input:
            img = self.transformer.transformer(img).flatten()
            data_in.append([img.tolist()])
        return data_in

    def forward(self, input_data):
        in_arg = self.converter(input_data)
        return self.network.forwardTest(in_arg)

    def forward(self, data, output_layer):
        #返回每种标签的概率，详见http://www.paddlepaddle.org/doc_cn/ui/predict/swig_py_paddle.html
        input = self.converter(data)
        self.network.forwardTest(input)
        output = self.network.getLayerOutputs(output_layer)
        return output[output_layer].mean(0)

    def predict(self, image=None, output_layer=None):
        assert isinstance(image, basestring)
        assert isinstance(output_layer, basestring)
        data = self.get_data(image)#读取图片
        prob = self.forward(data, output_layer)
        lab = np.argsort(-prob)#按降序排列,返回的是数组值的索引值
        logging.info("Label of %s is: %d", image, lab[0])

if __name__ == '__main__':
    image_size=28#图像大小
    crop_size=28#图像大小
    multi_crop=0#多次裁剪
    config="vgg.py"#网络配置文件
    output_layer="__fc_layer_1__"
    mean_path="data/batches/batches.meta"
    model_path="vgg_model/pass-00000/"#模型路径
    image="test.bmp"#要识别的图片路径
    use_gpu=0#是否使用GPU

    obj = ImageClassifier(train_conf=config,
                          model_dir=model_path,
                          resize_dim=image_size,
                          crop_dim=crop_size,
                          mean_file=mean_path,
                          use_gpu=use_gpu,
                          oversample=multi_crop)
    obj.predict(image, output_layer)
