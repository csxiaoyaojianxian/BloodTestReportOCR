# -*- coding: utf-8 -*-
"""
pandas 0.18.1
scikit-learn 0.18.1
matplotlib 1.5.3
numpy 1.11.1
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostClassifier

class_names_train2 = ['sex', 'age', 'WBC', 'RBC', 'BAS#', 'HGB', 'HCT', 'MCV',
                          'MCH', 'MCHC', 'RDW-CV', 'PLT', 'MPV', 'PCT', 'PDW', 'LYM#',
                          'LYM%', 'MONO', 'MONO%', 'NEU#', 'NEU%', 'EOS#', 'EOS%', 'BAS%',
                          'IG#', 'IG%', 'NRBC#', 'NRBC%', 'P-LCR']


def load_data():
    # 数据集已合并, 去掉了标签行
    # sex预处理: 男是1, 女是0

    df = pd.DataFrame(pd.read_csv('train2.csv', names=class_names_train2))
    df = df.convert_objects(convert_numeric=True)
    df = df.fillna(df.mean())

    # 去掉id, 分裂标签
    selected_names = [x for x in class_names_train2 if (x != 'sex' and x != 'age')]
    X_data = df[selected_names].as_matrix()
    y_data = df['sex'].as_matrix().astype(int)
    return X_data, y_data


def data_preprocess(X_data, y_data):
    # 按3:1分裂训练集/测试集
    X_train, X_test, y_train, y_test = \
        train_test_split(X_data, y_data, test_size=0.25)
    return X_train, X_test, y_train, y_test


def evalue(clf, X_test, y_test):
    """
    评估模型在测试集上的性能
    :param clf: 模型
    :param X_test: 测试集数据
    :param y_test: 测试集标记
    :return:
    """
    pd = clf.predict(X_test)

    correct_pairs = [(x, y) for (x, y) in zip(y_test, pd) if x == y]
    precision = float(len(correct_pairs)) / len(pd)

    print '准确率为: ' + str(precision)


def feature_select(clf, X_train, y_train, X_test):
    # 预训练
    clf.fit(X_train, y_train)
    
    # 评估特征
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("特征权值分布为: ")
    for f in range(X_train.shape[1]):
        print("%d. %s %d (%f)" % (f + 1, class_names_train2[indices[f]], indices[f], importances[indices[f]]))
    
    # 过滤掉权值小于threshold的特征
    model = SelectFromModel(clf, threshold=0.04, prefit=True)
    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)
    print '训练集和测试集的容量以及选择的特征数为: ', X_train_new.shape, X_test_new.shape
    # 返回压缩特征之后的训练集和测试集
    return X_train_new, X_test_new


if __name__ == '__main__':
    # 载入数据
    X_data, y_data = load_data()
    X_train, X_test, y_train, y_test = data_preprocess(X_data, y_data)

    # 使用adaboost
    clf = clf = AdaBoostClassifier()
    # 选择特征, 压缩数据
    X_train_compressed, X_test_compressed = feature_select(clf, X_train, y_train, X_test)
    
    # 使用选择的特征重新训练
    clf.fit(X_train_compressed, y_train)
    # 评估模型
    evalue(clf, X_test_compressed, y_test)
