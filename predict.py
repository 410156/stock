import numpy as np
import pandas as pd
import json
import os
import sys
import pickle
from random import randint
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time

stock_codes = pickle.load(open('./stock_number/ep.pkl','rb'))
all_col = pickle.load(open('selected_cols_for_total_data.pkl', 'rb'))
y = ['y_updown','y_point','y_percent']
x = []

for i in all_col:
    if i != y[0] and i!= y[1] and i!= y[2]:
        x.append(i)
files = os.listdir('./new_total_data')


for i in range(len(files)):
    with open(f'./new_total_data/{files[i]}', 'rb') as f:
        print(files[i])
        data = pickle.load(f)
        data = data.reset_index()
        data = data.drop(columns='index')

        start = time.time()
        for s in all_col:
            for j in range(len(data[s])):       
                if type(data[s][j])==str:
                    if ',' in data[s][j]:
                        data[s][j] = float(data[s][j].replace(',', ''))
        print(time.time()-start)
        data_y = pd.DataFrame()
        data_y['y_point'] = data[y[0]]
        data_y['y_point'] = pd.to_numeric(data_y['y_point'])
        data_x = data[x]
        for f in data_x.columns:
            data_x[f] = pd.to_numeric(data_x[f])
        labelencoder = LabelEncoder()
        data_x['stock_num'] = labelencoder.fit_transform(data_x['stock_num'])
        data_Dmatrix = xgb.DMatrix(data=data_x,label=data_y)
        x_train,x_test,y_train,y_test = train_test_split(data_x,data_y, train_size=0.8139,shuffle = False)
        
        y_train['y_point'] = pd.to_numeric(y_train['y_point'])
        y_train['y_point'][len(y_train)-1] = 0.0
        params = {
        'objective':'binary:logistic',
        'max_depth':4,
        'learing_rate':1.0
        }
        xgb_clf = XGBClassifier(**params)

        y_train = y_train.values
        y_test = y_test.values

        '''
        y_train[y_train==True] = 1
        y_train[y_train==False] = 0
        y_train[pd.isnull(y_train)] = 0

        y_test[y_test==True] = 1
        y_test[y_test==False] = 0
        y_test[pd.isnull(y_test)] = 0

        x_train = x_train.values
        x_test = x_test.values

        x_train = x_train[:-1, :]
        x_test = x_test[:-1, :]
        y_train = y_train[1:]
        y_test = y_test[1:]
        '''

        xgb_clf.fit(x_train,y_train)


        y_predict = xgb_clf.predict(x_test)

        print('accuracy:{0:0.4f}'.format(accuracy_score(y_test,y_predict)))
        report = classification_report(y_test, y_predict)
        print(report)
        break
        stock_code = files[i].replace('_merge.pkl', '')
        fo = open(f'./predict/{stock_code}_clf_replort.txt', 'w')
        fo.write(report)
        fo.close()
