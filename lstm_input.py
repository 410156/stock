import numpy as np
import pandas as pd
import pickle
from pandas import Series
import os
from datetime import datetime



def combine_files(files,all_col,full_x_array,full_y_array):


    for x in files:
        with open(f'./month_predict_input/{x}' , 'rb') as f:
            stock_code = x.replace('_month_fin_merge.pkl', '')
            print(stock_code)
            
            stock_data = pickle.load(f)
            stock_data = stock_data.reset_index().rename(columns={'index': 'Date'})

            start_date = datetime(year=2004,month=2,day=11)#記錄每個月的第一個開盤日
            one_month_day = 0#用來記算每個月的天數
            new_point = 0#記錄新月分第一個開盤日的收盤價
            pre_point = 0#記錄上一個月第一個開盤日的收盤價
            pre_day = 0#記錄前一個迴圈日期的日
            max_month_day = 24#一個月最多24個開盤日
            for date in stock_data['Date']:
                #若上一個回圈的日比這個回圈的日大代表已經到下一個月了,且date為新月份第一個日期
                if date.day<pre_day:
                    #把上一個月份的日期區間資料append到full_x_array
                    full_x_array = np.append(full_x_array,stock_data[stock_data['Date']<date][stock_data['Date']>=start_date][all_col].values)
                    #用這個月最後一開盤日數據補足剩餘不滿24天的數據
                    for null_date in range(max_month_day-one_month_day):
                        full_x_array = np.append(full_x_array,stock_data[stock_data['Date'] == full_x_array[-1]][all_col])
                    #start_date目前是上一個月的第一個開盤日
                    y_date = start_date
                    start_date = date
                    #新月份第一個開盤日的收盤價
                    new_point = stock_data[stock_data['Date'] == date]['close'].values
                   
                    if new_point-pre_point>0:
                        full_y_array = np.append(full_y_array,1)
                    elif new_point-pre_point==0:
                        full_y_array = np.append(full_y_array,0)
                    else:
                        full_y_array = np.append(full_y_array,-1)
                    #把新月份減舊月份收盤價放到舊月份第一個開盤日
                    full_y_array = np.append(full_y_array,y_date)
                    pre_point = new_point
                    one_month_day = 1
                    pre_day = date.day
                else:
                    one_month_day = one_month_day + 1
                pre_day = date.day

    print(full_x_array.shape)
    print(full_y_array.shape)

    return full_x_array, full_y_array



if '__main__' == __name__:

    files = os.listdir('./month_predict_input')
    all_col = pickle.load(open('selected_cols_for_total_data.pkl', 'rb'))
    all_col.append('Date')
    all_col = [s for s in all_col if s != 'y_updown' and s != 'y_percent' and s != 'y_point']

    full_x_array = np.empty((0,len(all_col)))
    full_y_array = np.empty((0,1))

    full_x_array,full_y_array = combine_files(files,all_col,full_x_array,full_y_array)

    pickle.dump(full_x_array,open(f'./monthfin_lstm_x_input.pkl','wb'))
    pickle.dump(full_y_array,open(f'./monthfin_lstm_y_input.pkl','wb'))

