# -*- coding: UTF-8 -*-
from paddle.trainer_config_helpers import *

is_predict = get_config_arg("is_predict", bool, False)

####################Data Configuration ##################
if not is_predict:
  data_dir='data/batches/'
  meta_path=data_dir+'batches.meta'

  '''
  mean_img_size,img_size图像大小
  num_classes分类类别数
  color图像有无颜色
  '''
  args = {'meta':meta_path,'mean_img_size': 28,
          'img_size': 28,'num_classes': 10,
          'use_jpeg': 1,'color': 0}

  #引用image_provider.py中的processData函数
  define_py_data_sources2(train_list=data_dir+"train.list",
                          test_list=data_dir+'test.list',
                          module='dataprovider',
                          obj='processData',
                          args=args)

######################Algorithm Configuration #############
settings(
    #批尺寸，一次训练多少数据
    batch_size = 128,
    #学习速率
    learning_rate = 0.1 / 128.0,
    #学习方式，详见http://www.paddlepaddle.org/doc/ui/api/trainer_config_helpers/optimizers.html
    learning_method = MomentumOptimizer(0.9),
    #权重衰减，防过拟合
    regularization = L2Regularization(0.0005 * 128)
)

#######################Network Configuration #############
#图片大小，通道数×长×宽
data_size=1*28*28
#分类数量
label_size=10
#关于layer,详见http://www.paddlepaddle.org/doc/ui/api/trainer_config_helpers/layers.html
img = data_layer(name='image',
                 size=data_size)
#small_vgg在trainer_config_helpers.network预定义
#关于网络详见http://www.paddlepaddle.org/doc/ui/api/trainer_config_helpers/networks.html
predict = small_vgg(input_image=img,
                    num_channels=1,#图像通道数，灰度图像为1
                    num_classes=label_size)

if not is_predict:
    lbl = data_layer(name="label", size=label_size)
    outputs(classification_cost(input=predict, label=lbl))
else:
    #预测网络直接输出最后一层的结果而不是像训练时以cost layer作为输出
    outputs(predict)

