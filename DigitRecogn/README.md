### 神经网络实现手写字符识别系统

 - BP神经网络
 - 输入层维数 400
 - 隐藏层神经元 15
 - 输出层维数 10
 - 学习率 0.1
 - 激活函数 sigmoid
 - 参数保存在 nn.json

#### 环境配置(如果在本地运行)
 - 系统: ubuntu 14.04 64位

```
# 安装pip
sudo apt-get install python-pip

# 用pip安装numpy和scipy, 使用科大镜像加速
pip install --user numpy scipy -i https://pypi.mirrors.ustc.edu.cn/simple

# 如果上一步安装失败就使用ubuntu的包管理器试试
sudo apt-get install python-numpy python-scipy

# 安装sklearn, neural_network_design.py需要调用它做交叉验证
pip install -U scikit-learn -i https://pypi.mirrors.ustc.edu.cn/simple

# 如果在服务器上运行，修改ocr.js里的HOST为服务器的地址，如http://2016.mc2lab.com

```


#### 运行

1. 下载图像和标签数据


        wget http://labfile.oss.aliyuncs.com/courses/593/data.csv
        wget http://labfile.oss.aliyuncs.com/courses/593/dataLabels.csv


2. 训练模型

        python neural_network_design.py

3. 创建服务器

        python -m SimpleHTTPServer 3000

4. 加载服务器

        python server.py

5. 访问

        localhost:3000


* 实现指导见https://www.shiyanlou.com/courses/593
