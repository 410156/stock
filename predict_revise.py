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
files = os.listdir('./predict_input')


for i in range(len(files)):
    with open(f'./predict_input/{files[i]}', 'rb') as f:
        data = pickle.load(f)

        data_y = pd.DataFrame()
        data_y['y_updown'] = data[y[0]]
        data_y['y_updown'] = pd.to_numeric(data_y['y_updown'])
        data_x = data[x]
        for f in data_x.columns:
            data_x[f] = pd.to_numeric(data_x[f])
        labelencoder = LabelEncoder()
        data_x['stock_num'] = labelencoder.fit_transform(data_x['stock_num'])
        data_Dmatrix = xgb.DMatrix(data=data_x,label=data_y)
        x_train,x_test,y_train,y_test = train_test_split(data_x,data_y, train_size=0.8139,shuffle = False)
        
        y_train['y_updown'] = pd.to_numeric(y_train['y_updown'])
        y_train['y_updown'][len(y_train)-1] = 0.0
        
        params = {
        'objective':'binary:logistic',
        'max_depth':4,
        'learing_rate':1.0
        }
        xgb_clf = XGBClassifier(**params)

        xgb_clf.fit(x_train,y_train)

        y_predict = xgb_clf.predict(x_test)

        print('accuracy:{0:0.4f}'.format(accuracy_score(y_test,y_predict)))
        report = classification_report(y_test, y_predict)
        print(report)
        stock_code = files[i].replace('_merge.pkl', '')
        fo = open(f'./predict/{stock_code}_clf_report.txt', 'w')
        fo.write(report)
        fo.close()
