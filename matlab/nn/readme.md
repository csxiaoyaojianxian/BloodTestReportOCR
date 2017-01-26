# matlab安装

学校提供正版matlab，可用校园VPN在http://ms.ustc.edu.cn/zbh.php下载

安装路径不能有中文

# 数据

我把数据整理成mat格式文件

链接：http://pan.baidu.com/s/1hr95FEW 密码：6ibz

 train_input_transpose
 
 train_output_transpose
 
 predict_input_transpose 
 
 predict_output_transpose
 
 这几个是转置后的数据，因为matlab的nntool把每一列看做一个单元，所以需要转置。
 
# 程序

 network_hit139
 
 一个已经训练好的网络模型，准确率69.50%  

 create_nn.m
 
 创造一个新的神经网络并训练
 
 test_nn.m
 
 神经网络测试
 
 也可以在命令行中输入“nntool”，导入 network_hit139后进行编辑。
 
 具体见
 
 http://blog.csdn.net/qq_28591171/article/details/54172146#1matlab神经网络