# -*- coding: utf-8 -*-
"""
pandas 0.18.1
scikit-learn 0.18.1
matplotlib 1.5.3
numpy 1.11.1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

# 使用了预处理的第二组数据集
class_names_train2 = ['sex','age','WBC','RBC','BAS#','HGB','HCT','MCV',
                      'MCH','MCHC','RDW-CV','PLT','MPV','PCT','PDW','LYM#',
                      'LYM%','MONO','MONO%','NEU#','NEU%','EOS#','EOS%','BAS%',
                      'IG#','IG%','NRBC#','NRBC%','P-LCR']


def load_data():
    # 数据集已合并, 去掉了标签行, sex预处理为数字
    df = pd.DataFrame(pd.read_csv('train2.csv', names=class_names_train2))
    # 转化为字符串
    df = df.convert_objects(convert_numeric=True)
    # 使用平均值填充缺失值
    df = df.fillna(df.mean())
    return df


def split_data(df, low, high):
    """
    :param df: 输入的dataframe
    :param low: 截取区间的低阈值
    :param high: 截取区间的高阈值(不包含)
    :return: 截取的dataframe
    """
    df_lowcut = df[df['age'] >= low]
    df_cut = df_lowcut[df_lowcut['age'] < high]

    selected_names = [x for x in class_names_train2 if (x != 'age' and x != 'sex')]
    x_data = df_cut[selected_names].as_matrix()
    y_data = df_cut['age'].as_matrix()
    # 用平均值填充nan
    def fill_nan(np_array):
        col_mean = np.nanmean(np_array, axis=0)
        nan_ids = np.where(np.isnan(np_array))
        np_array[nan_ids] = np.take(col_mean, nan_ids[1])
        return np_array

    x_data = fill_nan(x_data)
    print 'x有没有nan值:', np.any(np.isnan(x_data))
    print 'y有没有nan值:', np.any(np.isnan(y_data))

    return x_data, y_data


def draw(labels, prediction):
    """
    绘制折线图比较结果
    :param labels: 1维numpy数组
    :param prediction: 1维numpy数组
    :return:
    """
    result = []
    for i in range(labels.shape[0]):
        result.append([labels[i], prediction[i]])

    # 将年龄按照大小排序
    result = sorted(result, key=lambda x: x[0])
    labels = [row[0] for row in result]
    prediction = [row[1] for row in result]

    plt.plot(labels, label='labels')
    plt.plot(prediction, label='predict')
    plt.legend(loc='upper left')
    plt.show()


# 评估测试集
def evalue(clf, X_test, y_test):
    pd = clf.predict(X_test)
    
    delta = [x1 - x2 for (x1, x2) in zip(y_test, pd)]
    correct_indices = [x for x in delta if abs(x) < 5]
    precision = float(len(correct_indices)) / len(pd)
    
    print '准确率为: ' + str(precision)
    draw(y_test, pd)


def feature_select(clf, X_train, y_train, X_test):
    # 预训练
    print '特征选择预训练中...'
    clf.fit(X_train, y_train)
    
    # 评估特征
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("特征权值分布为: ")
    for f in range(X_train.shape[1]):
        print("%d. %s %d (%f)" % (f + 1, class_names_train2[indices[f]], indices[f], importances[indices[f]]))
    
    # 过滤掉权值小于threshold的特征
    model = SelectFromModel(clf, threshold=0.01, prefit=True)
    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)
    print '训练集和测试集的容量以及选择的特征数为: ', X_train_new.shape, X_test_new.shape
    # 返回压缩特征之后的训练集和测试集
    return X_train_new, X_test_new


if __name__ == '__main__':
    #载入数据
    df = load_data()
    x1, y1 = split_data(df, 0, 25)
    x2, y2 = split_data(df, 25, 60)
    x3, y3 = split_data(df, 60, 80)

    def test_data(X_data, y_data):
        # 按9:1分裂训练集/测试集
        X_train, X_test, y_train, y_test = \
            train_test_split(X_data, y_data, test_size=0.1, random_state=0)
        # 使用随机森林
        clf = RandomForestRegressor(max_features=None, n_estimators=20, max_depth=None)
        # 特征选择
        X_train_compressed, X_test_compressed = feature_select(clf, X_train, y_train, X_test)
        # 使用提取的特征重新训练
        clf.fit(X_train_compressed, y_train)
        # 评估训练集效果
        evalue(clf, X_train_compressed, y_train)
        # 评估测试集效果
        evalue(clf, X_test_compressed, y_test)

    test_data(x1, y1)
    test_data(x2, y2)
    test_data(x3, y3)
