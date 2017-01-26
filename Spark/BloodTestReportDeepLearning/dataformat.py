# -*- coding: cp936 -*-
#生成LabeledPoints类型数据，格式如下：
#label,factor1 factor2 .... factorn
#第一列为类别标签，后面以空格隔开的为特征（因子）


import csv

reader = csv.reader(file('./data_set.csv', 'rb'))
output1 = open('LabeledPointsdata_age.txt', 'w')
output2 = open('LabeledPointsdata_sex.txt', 'w')

flag = 0
row = 0

for line in reader:
    row = row + 1
    if 1 == row:
        continue

    column = 0
    for c in line:
        column = column + 1
        if 1 == column:
            continue
        if 2 == column:
            if "男" == c:
                outputline2 = "0,"
            else:
                outputline2 = "1,"
            continue
        if 3 == column:
            outputline1 = c + ","
        else:
            if "--.--"==c:
                flag = 1
                break
            else:
                outputline1 += (c + " ")
                outputline2 += (c + " ")
    if 0 == flag:
        outputline1 += '\n'
        outputline2 += '\n'
    else:
        flag = 0
        continue
    print(column)
    output1.write(outputline1)
    output2.write(outputline2)
output1.close()
output2.close()
print('Format Successful!')