#!usr/bin/env python3
# -*- coding: utf-8 -*-

<<<<<<< HEAD

from bs4 import BeautifulSoup as bs
from datetime import datetime
=======
import numpy as np
import pandas as pd
>>>>>>> c934826dac45ca6bdfa018f894e102136913950c
import json
import numpy as np
import os
import pandas as pd
from pandas import Series
import pickle
from random import randint
import re
import requests
import sys
<<<<<<< HEAD
import statistics as st
from scipy import stats 
from talib import abstract
import time

               
def merge_data(technical, fincial_report):
    '''
    input fincial_report : 
=======
import time
from random import randint
from bs4 import BeautifulSoup as bs
from datetime import datetime
from pandas import Series
from talib import abstract



# data_kd -> stock_indices
# chart -> ?


def merge_data(data_kd, chart):

    ''' input chart :
>>>>>>> c934826dac45ca6bdfa018f894e102136913950c

                1091231 1081231
    cash-value     aa       cc
    cash-rate      bb       dd


        input technical :
                K D High Low
    2014-01-01  a b  c    d
    2014-01-02  e f  g    g

        output :
                K D High Low cash-value cash-rate
    2014-01-01  a b  c    d      aa      cc
    2014-01-02  e f  g    g      bb     dd
<<<<<<< HEAD
    ''' 
    for date in technical['Date']: # for each trading date
        print(date)
=======
     '''
    for date in data_kd['Date']: # for each trading date
>>>>>>> c934826dac45ca6bdfa018f894e102136913950c
        # sequentially search for the corresponding report date
        for i in range(1, len(fincial_report.columns)): # for each report date
            if fincial_report.columns[i-1]<date and date<=fincial_report.columns[i]:
                fincial_report[date] = fincial_report[fincial_report.columns[i-1]]
                break #avoid overwritting
            else:
<<<<<<< HEAD
                fincial_report[date] = ''*len(fincial_report.index)
    technical = technical.set_index('Date')
    fincial_report = fincial_report.T
    merge = pd.concat([fincial_report, technical], axis=1)
    return merge
=======
                chart[date] = ''*len(chart.index)#沒有在區間內設空值
    data_kd = data_kd.set_index('Date')
    chart = chart.T
    total_data = pd.concat([chart, data_kd], axis=1)

    return total_data
>>>>>>> c934826dac45ca6bdfa018f894e102136913950c




#! z -> change

def load_data(path, stock_codes):
<<<<<<< HEAD
    files = sorted(os.listdir(path))
=======

    files = sorted(os.listdir(path))

>>>>>>> c934826dac45ca6bdfa018f894e102136913950c
    total_data = {}
    for _file in files:
        with open(path+_file, 'rb') as f:
            data = json.load(f)
<<<<<<< HEAD
=======
            print(z)
            print(data['2414'])
>>>>>>> c934826dac45ca6bdfa018f894e102136913950c
            for stock in stock_codes:
                if stock not in total_data.keys():
                    total_data[stock] = []
                if stock not in data.keys():
                    total_data[stock].append([datetime.strptime(_file.strip('.json'), '%Y-%m-%d'), stock, np.nan, np.nan, np.nan, np.nan])
                else:
                    try :
                        total_data[stock].append([datetime.strptime(_file.strip('.json'), '%Y-%m-%d'), stock, float(data[stock]['open']), float(data[stock]['high']), float(data[stock]['low']), float(data[stock]['close'])])
                    except:
                        total_data[stock].append([datetime.strptime(_file.strip('.json'), '%Y-%m-%d'), stock, np.nan, np.nan, np.nan, np.nan])
    for code in total_data.keys():
        total_data[code] = np.array(total_data[code])

    to_df = {}
    for key in total_data.keys():
        to_df[key] = pd.DataFrame(total_data[key], columns=['Date', 'stock_num', 'open', 'high', 'low', 'close'])

    return to_df


class Add_chart:

    def send_request(self,code,year,season,url):
        stock_dic = {
            'encodeURIComponent': '1',
            'step': '1',
            'firstin': '1',
            'off': '1',
            'keyword4': '',
            'code1': '',
            'TYPEK2': '',
            'checkbtn': '',
            'queryName': 'co_id',
            'inpuType': 'co_id',
            'TYPEK': 'all',
            'isnew': 'fale',
            'co_id': code,
            'year':year,
            'season':season
        }

        headers = {'content-type': 'charset=utf8'}
        res = requests.post(url, data = stock_dic, headers=headers)
        soup = bs(res.text,'lxml')
        return soup

    def financial_crawler(self,url,stock_code_list):
        print(stock_code_list)
        total_df=pd.DataFrame()
        for code in stock_code_list:
            print(code)
            intial = 1
            for year in range(102,110):
                for season in range(1,5):

                    soup = self.send_request(code,year,season,url)


                    both = soup.find_all('td',class_=['even', 'odd'])

                    content = [i.text.replace(' ', '').replace('\u3000', '') for i in both]
                    col_num = content[:10].count('')+1#算出此季報共有多少個日期,有5,7,9三種
                    dates = [soup.find_all('th',class_='tblHead')[i].getText().lstrip() for i in range(3,3+int(col_num/2))]#找出此季報的日期
                    new_dates = ['欄位名稱']+ [dates[int(i/2)]+'_value' if i%2==0 else dates[int(i/2)]+'_rate' for i in range(col_num-1)]#把欄位名稱加兩種日期放進new_data,兩種日期為加上'-1'與'-2','-1'是值'-2'是rate


                    content = np.array(content).reshape(-1, col_num)#把data reshape為日期為columns 資料名稱為index的形狀
                    content = content[content[:, 1]!='']
                    print(content.shape)
                    exit()
                    df = pd.DataFrame(content, columns=new_dates)
                    df = df.iloc[0:-1,0:3]#只存第一個日期的資料
                    if intial == 1:
                        total_df = df
                    else:
                        total_df=pd.merge(total_df,df)
                    intial = 0
                    time.sleep(randint(2, 6))
                    print(season)

            total_df = total_df.set_index('欄位名稱')
            total_df.index = total_df.index+'-value'
            chart_rate = pd.DataFrame()
            for column in total_df.columns:
                if column[-2] == 'u':
                    total_df = total_df.rename(columns={column:column[0:-6]})
                else:
                    chart_rate[column[0:-5]] = total_df[column]
                    total_df = total_df.drop(columns = column)
            chart_rate.index = [i[0:-6]+'-rate' for i in chart_rate.index]
            total_df = total_df.append(chart_rate)
            total_df.columns = [datetime.strptime(str(int(total_df.columns[i][0:3])+1911)+'-'+total_df.columns[i][4:6]+'-'+total_df.columns[i][7:9],"%Y-%m-%d") for i in range(len(total_df.columns))]
            return total_df



class Generate_index:

    def kd(self,target):
        target['RSV'] = 0
        target['RSV'] = 100*(target['close']-target['low'].rolling(9).min())/(target['high'].rolling(9).max()-target['low'].rolling(9).min())
        target['K'] = target['RSV']*9/10 
        target['K']= target['K'].shift(periods=1)*2/3 + target['RSV']/3
        target['D'] = target['K']
        target['D']= target['D'].shift(periods=1)*2/3 + target['K']/3
       
        return target

    def ema(self,target):
        target['EMA(12)'], target['EMA(26)'], target['DIF'] =  target['close']*2/13, target['close']*2/27, target['close']*2/13-target['close']*2/27
        target['EMA(12)']=(target['EMA(12)'].shift(periods=1)*11+2*target['close'])/13
        target['EMA(26)']=(target['EMA(26)'].shift(periods=1)*25+2*target['close'])/27
        target['DIF']=target['EMA(12)']-target['EMA(26)']
        return target

    def macd(self,target):
        x=9
        target['MACD']=target['DIF']*2/(x+1)
        target['MACD']=(target['MACD'].shift(periods=1)*(x-1)+target['DIF']*2)/(x+1)
        return target

    def y(self,target):
<<<<<<< HEAD
        target['y_point']=target['close'].shift(periods=-1)-target['close']
        target['y_point'][pd.isnull(target['y_point'])] = 0

        target['y_percent'] = target['y_point']/target['close']
        target['y_percent'][pd.isnull(target['y_percent'])] = 0

        target['y_updown'] = 0
        for i in range(1,len(target['y_point'])):
            if target['y_point'][i]>0:
                target['y_updown'][i] = 1
            elif target['y_point'][i]<0:
                target['y_updown'][i] = 0
            else:  
                target['y_updown'][i] = 0
        
=======
        target['y_updown']=target['close']-target['close'].shift(periods=1)
        target['y_percent'] = (target['close']-target['close'].shift(periods=1))/target['close']
        #target['y_point'] = [1 if target['y_updown'][i]>0  else -1 if target['y_updown'][i]<0  else 0 for i in range(1,len(target['y_updown']))]
        target['y_point'] = 0
        for i in range(1,len(target['y_updown'])):
            if target['y_updown'][i]>0:
                target['y_point'][i] = 1

            elif target['y_updown'][i]<0:
                target['y_point'][i] = -1

            else:#if target['y_updown'][i]==0:
                target['y_point'][i] = 0

        #    else:
        #        target['y_point'][i] = np.NaN
>>>>>>> c934826dac45ca6bdfa018f894e102136913950c
        return target

    def ma(self,target):
        target['5MA']=(target['close']+target['close'].shift(periods=1)+target['close'].shift(periods=2)+target['close'].shift(periods=3)+target['close'].shift(periods=4))/5
        target['20MA'] = target['close']
        for i in range(1,20):
            target['20MA'] = target['20MA'] + target['close'].shift(periods=i)
        target['20MA'] = target['20MA']/20

        target['60MA'] = target['close']
        for i in range(1,60):
            target['60MA'] = target['60MA'] + target['close'].shift(periods=i)
        target['60MA'] = target['60MA']/60

        target['5MA over 20MA'] = False
        for i in range(1,len(target['20MA'])):
            if target['5MA'][i-1]<target['20MA'][i-1] and target['5MA'][i]>target['20MA'][i] and target['20MA'][i]!=np.NaN:
                target['5MA over 20MA'][i] = True
            else:
<<<<<<< HEAD
                target['5MA over 20MA'][i] = False   
        target['20MA over 5MA'] = False
=======
                target['5MA over 20MA'][i] = 0
        target['20MA over 5MA'] = 0
>>>>>>> c934826dac45ca6bdfa018f894e102136913950c
        for i in range(1,len(target['20MA'])):
            if target['5MA'][i-1]>target['20MA'][i-1] and target['5MA'][i]<target['20MA'][i] and target['20MA'][i]!=np.NaN:
                target['20MA over 5MA'][i] = True
            else:
<<<<<<< HEAD
                target['20MA over 5MA'][i] = False   
=======
                target['20MA over 5MA'][i] = 0

       #target['20MA over 5MA']=[1 if ttarget['5MA'][i-1]>target['20MA'][i-1] and target['5MA'][i]<target['20MA'][i] and target['20MA']!=np.NaN else 0 for i in range(1,len(target['20MA']))]
>>>>>>> c934826dac45ca6bdfa018f894e102136913950c
        return target


    def generate(self,data,key):
        data[key] = self.kd(data[key])
        data[key] = self.ema(data[key])
        data[key] = self.macd(data[key])
        data[key] = self.y(data[key])
        data[key] = self.ma(data[key])
        return data[key]


<<<<<<< HEAD

if '__main__' == __name__:

    
    # generate pretict_revise.py's input
    stock_codes = pickle.load(open('./stock_number/ep.pkl','rb'))[7:-1]
    #all_col = pickle.load(open('selected_cols_for_total_data.pkl', 'rb'))

    for i in stock_codes:
        print(i)
        if(i!='3055' and i!='6281' and i!='6776' and i!='2414'):
            df = pickle.load(open(f'./kd_data/{i}_technical.pkl','rb'))
            print(df)
            chart = pickle.load(open(f'./final_chart/{i}_chart.pkl','rb'))   
            chart = chart.T 
            all_col = chart.columns
            for s in all_col:
                for j in range(len(chart[s])):       
                    if type(chart[s][j])==str:
                        if ',' in chart[s][j]:
                            chart[s][j] = float(chart[s][j].replace(',', ''))
            chart = chart.T
            print(chart)
            final_data = merge_data(df, chart)
            for x in range(len(final_data['y_point'])):
              if np.isnan(final_data['y_point'][x]):
                  final_data['y_point'][x] = 0
            for x in range(len(final_data['y_percent'])):
               if np.isnan(final_data['y_percent'][x]):
                   final_data['y_percent'][x] = 0
            for x in range(len(final_data['y_updown'])):
                if np.isnan(final_data['y_updown'][x]):
                    final_data['y_updown'][x] = 0
            pickle.dump(final_data,open(f'./predict_input/{i}_merge.pkl','wb'))
            print(final_data)
=======
if '__main__' == __name__:

    stock_codes = pickle.load(open('./stock_number/ep.pkl','rb'))[17:-1]
    print(len(stock_codes))

    #stock_codes=[]
    #for i in new:
    #    stock_codes.append(str(i))
    #print(stock_codes)
    path = '/home/db/stock_resource_center/resource/twse/json/'
    #stock_codes = ['2414']
    #print(stock_codes)
    #selected_stock = load_data(path, stock_codes)
    #exit()
    #add_index = Generate_index()
    #for code in stock_codes:
        #print(code)
        #df = add_index.generate(selected_stock, code) #-> ./data/8112.pkl 有kd
        #df = pickle.dump(df,open(f'./kd_data/{code}_technical.pkl','wb'))
    #exit()
    
    df = pickle.load(open('./kd_data/2347_technical.pkl','rb'))
    
    print(df)#.iloc[-200:-180,:])
    
>>>>>>> c934826dac45ca6bdfa018f894e102136913950c
    exit()


    # generate technical data
    tmp = ['']
    stock_codes = pickle.load(open('./stock_number/ep.pkl','rb'))[1:-1]
    for i in stock_codes:
        print(i)
        if(i!='3055' and i!='6281' and i!='6776'):
            path = '/home/db/stock_resource_center/resource/twse/json/'
            tmp[0] = i
            selected_stock = load_data(path, tmp)
            add_index = Generate_index()
            df = add_index.generate(selected_stock, tmp[0])
            print(df)
            exit()
            #pickle.dump(df,open(f'./kd_data/{code}_technical.pkl','wb'))
    exit()

    # generate fincial report  data
    url = "https://mops.twse.com.tw/mops/web/t164sb03"
    get_chart = Add_chart()
        

