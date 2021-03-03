import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import sys
import pickle
from pandas import Series
######




path_kd = '/home/qqqqqq/stock/include_KDdata/'
_file_kd = os.listdir(path_kd)
for z in _file_kd:
    with open(path_kd+z, 'rb') as f:
        data_kd = pickle.load(f)


path_chart = '/home/qqqqqq/stock/chart/'
_file_chart = os.listdir(path_chart)
for z in _file_chart:
    with open(path_chart+z, 'rb') as f:
        data_chart = pickle.load(f)

data_1 = pd.DataFrame()
data_2 = pd.DataFrame()
for column in data_chart.columns:
    if column[-1] == '1':
        data_1[column] = data_chart[column]
    else:
        data_2[column] = data_chart[column]

data_1 = data_1.stack().unstack(level=0)
data_2 = data_2.stack().unstack(level=0)

data_1_index = []

for i in range(0,len(data_1.index)): 
    data_1_index.append(datetime.strptime(str(int(data_1.index[i][0:3])+1911)+'-'+data_1.index[i][4:6]+'-'+data_1.index[i][7:9],"%Y-%m-%d"))

data_1.index = data_1_index 
data_1_column = []

for i in range(0,len(data_1.columns)):
    data_1_column.append(data_1.columns[i]+'-value')

data_1.columns = Series(data_1_column)

data_2_index=[]

for i in range(0,len(data_2.index)): 
    data_2_index.append(datetime.strptime(str(int(data_2.index[i][0:3])+1911)+'-'+data_2.index[i][4:6]+'-'+data_2.index[i][7:9],"%Y-%m-%d"))

data_2.index = data_2_index 

data_2_column = []

for i in range(0,len(data_2.columns)):
    data_2_column.append(data_2.columns[i]+'-rate')

data_2.columns = Series(data_2_column)

data_1 = data_1.reset_index()
data_2 = data_2.reset_index()

sub_data_1 = pd.merge(data_1,data_2)

sub_data_1 = sub_data_1.set_index('index')
sub_data_1 = sub_data_1.stack().unstack(level=0)
sub_data_1 = sub_data_1.reset_index()
sub_data_1column=[]

sub_data_2 = pd.DataFrame(index = sub_data_1['index'])
d=0
for col_2 in data_kd['Date']:
    for col_1 in range(1,len(sub_data_1.columns)):
        if col_1 == 1:
            sub_data_2[col_2] = ''*len(sub_data_1.index)
        else:                
            if sub_data_1.columns[col_1-1]<col_2 and col_2<=sub_data_1.columns[col_1]: 
               d+=1
               for i in range(0,len(sub_data_2[col_2])):
                  for j in  range(0,len(sub_data_1.iloc[:,col_1] )):
                      if i==j:
                          sub_data_2[col_2][i] = sub_data_1.iloc[:,col_1][j]
               break             
            else:
                sub_data_2[col_2] = ''*len(sub_data_1.index)


sub_data_2 = sub_data_2.stack().unstack(level=0)            
sub_data_2 = sub_data_2.reset_index()

a = sub_data_2[0:-1]
b = data_kd[-len(sub_data_2.iloc[:,2]):-1]
a_column = ['Date']
for i in range(1,len(a.columns)):
    a_column.append(a.columns[i])
a.columns = a_column
total_data = pd.merge(a,b)
total_date = total_data.set_index('Date')
for i in total_data['現金及約當現金-value']:
    if i!='' :
        print(i)
print(total_data)
pickle.dump(total_data, open(f'./merge_data/8112.pkl' ,'wb'))


