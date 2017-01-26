#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:19:21 2016

@author: zhao
"""
import random
import numpy as np
import pandas as pd

class Traindata:
    def __init__(self):
        self.df = pd.read_csv('trainurl', index_col = 0)
        #将性别转化为2维矩阵,行代表病人id，列代表性别，为男则第一列置1,女则第二列置1
        self.gender = np.zeros((1858,2))
        for i in range(1858):
            if self.df.iloc[i,0]==1:
                self.gender[i,0]=1
            else:
                self.gender[i,1]=1
        self.age = self.df.loc[1:,['age']]
        #将26项指标转换为26列的矩阵
        self.parameter = self.df.loc[1:,['WBC','RBC','HGB','HCT','MCV','MCH','MCHC','ROW','PLT','MPV','PCT','PDW','LYM','LYM%','MON','MON%','NEU','NEU%','EOS','EOS%','BAS','BAS%','ALY','ALY%','LIC','LIC%']]
        self.parameter = np.array(self.parameter)
    #可以返回随机的n个数据
    def next_batch_gender(self,n):
        lable = np.zeros((n,2))
        para = np.zeros((n,26))
        for i in range(n):
            k=random.randint(0, 1858)
            if self.gender[k,0]==1:
                lable[i,0]=1
            else:
                lable[i,1]=1
            para[0] = self.parameter[k]
        return para,lable

    def next_batch_age(self,n):
        para = np.zeros((n,26))
        for i in range(n):
            k=random.randint(0, 1858)
            if(i==0):
                age = pd.DataFrame([self.age.iloc[k]])
            else:
                age.append(self.age.iloc[k])
            para[0] = self.parameter[k]
        return para,age
        

       
    
