# 利用scikit-learn预测病人性别以及年龄

## 环境配置(Ubuntu 14.04或以上版本)

```
sudo apt-get install python-numpy cython python-scipy python-matplotlib
pip install -U scikit-learn(如果不行就加sudo)
pip install pandas
```

## 使用
1. 下载预处理过的数据集

```
chmod +x download.sh
./download.sh
```

2. 预测

```
python gender_predict.py
python age_predict.py
```