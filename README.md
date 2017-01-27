# 对血常规检验报告的OCR识别、深度学习与分析

* 将血常规检验报告的图片识别出年龄、性别及血常规检验的各项数据
    * 图片上传页面，提交的结果是图片存储到了mongodb数据库得到一个OID或到指定目录到一个path
    * 图片识别得到一个json数据存储到了mongodb数据库得到一个OID，[json数据](https://coding.net/u/mengning/p/np2016/git/blob/master/BloodTestReportOCR/bloodtestdata.json)
       * 自动截取目标区域，已经能不同旋转角度的图片自动准备截取目标区域，但对倾斜透视的图片处理效果不佳,[具体用法](https://coding.net/u/mengning/p/np2016/git/blob/master/BloodTestReportOCR/README.md)
       * 预处理，比如增加对比度、锐化
       * 识别
           
    * 识别结果页面，上部是原始图片，下部是一个显示识别数据的表格，以便对照识别结果
* 学习血常规检验的各项数据及对应的年龄性别
* 根据血常规检验的各项数据预测年龄和性别

## Links

* [我的博客](http://www.csxiaoyao.com/blog/2017/01/01/ustc-np2016%E8%AF%BE%E7%A8%8B%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/)
