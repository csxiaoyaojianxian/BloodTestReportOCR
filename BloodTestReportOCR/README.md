
# 血常规检验报告OCR



## 运行环境

```
# 安装numpy,
sudo apt-get install python-numpy # http://www.numpy.org/
# 安装opencv
sudo apt-get install python-opencv # http://opencv.org/

##安装OCR和预处理相关依赖
sudo apt-get install tesseract-ocr
sudo pip install pytesseract
sudo apt-get install python-tk
sudo pip install pillow

# 安装Flask框架、mongo
sudo pip install Flask
sudo apt-get install mongodb # 如果找不到可以先sudo apt-get update
sudo service mongodb started
sudo pip install pymongo
```

## 运行

```
cd  BloodTestReportOCR
python view.py # upload图像,在浏览器打开http://yourip:8080

```

## view.py 

Web 端上传图片到服务器，存入mongodb并获取oid，稍作修整，希望能往REST架构设计，目前还不完善；
前端采用了vue.js, mvvm模式。写了两个版本，一个是index.html无插件，另一个使用了bootstrap-fileinput插件，有点问题；

## imageFilter.py
对图像透视裁剪和OCR进行了简单的封装，以便于模块间的交互，规定适当的接口
```    
    imageFilter = ImageFilter() # 可以传入一个opencv格式打开的图片
   
    num = 22
    print imageFilter.ocr(num)
```

#### ocr函数 - 模块主函数返回识别数据

用于对img进行ocr识别，他会先进行剪切，之后进一步做ocr识别，返回一个json对象
如果剪切失败，则返回None
@num 规定剪切项目数

#### perspect函数做 - 初步的矫正图片

用于透视image，他会缓存一个透视后的opencv numpy矩阵，并返回该矩阵
透视失败，则会返回None，并打印不是报告
@param 透视参数

* 关于param

参数的形式为[p1, p2, p3 ,p4 ,p5]。
p1,p2,p3,p4,p5都是整型，其中p1必须是奇数。

p1是高斯模糊的参数，p2和p3是canny边缘检测的高低阈值，p4和p5是和筛选有关的乘数。

如果化验报告单放在桌子上时，有的边缘会稍微翘起，产生比较明显的阴影，这种阴影有可能被识别出来，导致定位失败。
解决的方法是调整p2和p3，来将阴影线筛选掉。但是如果将p2和p3调的比较高，就会导致其他图里的黑线也被筛选掉了。
参数的选择是一个问题。
我在getinfo.default中设置的是一个较低的阈值，p2=70,p3=30，这个阈值不会屏蔽阴影线。
如果改为p2=70,p3=50则可以屏蔽，但是会导致其他图片识别困难。

就现在来看，得到较好结果的前提主要有三个
 - 化验单尽量平整
 - 图片中应该包含全部的三条黑线
 - 图片尽量不要包含化验单的边缘，如果有的话，请尽量避开有阴影的边缘。

#### filter函数 - 过滤掉不合格的或非报告图片

返回img经过透视过后的PIL格式的Image对象，如果缓存中有PerspectivImg则直接使用，没有先进行透视
过滤失败则返回None
@param filter参数


#### autocut函数 - 将图片中性别、年龄、日期和各项目名称数据分别剪切出来

用于剪切ImageFilter中的img成员，剪切之后临时图片保存在out_path，
如果剪切失败，返回-1，成功返回0
 @num 剪切项目数
 @param 剪切参数
 
剪切出来的图片在BloodTestReportOCR/temp_pics/ 文件夹下

函数输出为data0.jpg,data1.jpg......等一系列图片，分别是白细胞计数，中性粒细胞记数等的数值的图片。

#### classifier.py

用于判定裁剪矫正后的报告和裁剪出检测项目的编号

#### imgproc.py 
将识别的图像进行处理二值化等操作，提高识别率
包括对中文和数字的处理

#### digits
将该文件替换Tesseract-OCR\tessdata\configs中的digits
