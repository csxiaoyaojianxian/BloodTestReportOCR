# -*- coding:utf-8 -*-
import csv as cv
import numpy as np
import re
csv_file1_object = cv.reader(open('table1.csv','rb'))
csv_file2_object = cv.reader(open('table2.csv','rb'))
csv_file3=open('table3.csv','wb')
csv_file3_object=cv.writer(csv_file3)
head1 = csv_file1_object.next()
head2 = csv_file2_object.next()
data_2=[]
data2=[]
for row in csv_file2_object:
	#print len(row[1])
	if len(row[1])<10 and int(row[0])<=26:
		data_2.append(row)
	else:
		pass

data2=np.array(data_2)
col=0
data2=data2[np.argsort(data2[:,col])]


csv_file3_object.writerow(['id','sex','age','WBC','RBC','HGB','HCT','MCV','MCH'\
	,'MCHC','RDW','PLT','MPV','PCT','PDW','LYM','LYM%','MON','MON%','NEU','NEU%','EOS'\
	,'EOS%','BAS','BAS%','ALY','ALY%','LIC','LIC%'])
i=1
for row in csv_file1_object:
	right_only_stats= data2[(data2[0::,2]==row[2] ) & (data2[0::,3]==row[3]),1]

	#right_only_stats=np.column_stack((right_only_stats,np.array([row[0],row[1]])))
	right_only_stats=np.insert(right_only_stats,0,values=i,axis=None)
	i=i+1
	right_only_stats=np.insert(right_only_stats,1,values=row[0],axis=None)
	right_only_stats=np.insert(right_only_stats,2,values=row[1],axis=None)
	#right_only_stats= data2[0::,2]==row[2]
	if len(right_only_stats)==29:
		csv_file3_object.writerow(right_only_stats)