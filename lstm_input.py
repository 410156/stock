import numpy as np
import pandas as pd
import pickle
from pandas import Series
import os
from datetime import datetime



def combine_files(files,all_col,full_x,full_y):


    for x in files:
        with open(f'./month_predict_input/{x}' , 'rb') as f:
            stock_code = x.replace('_month_fin_merge.pkl', '')
            print(stock_code)
            
            stock_data = pickle.load(f)
            stock_data = stock_data.reset_index().rename(columns={'index': 'Date'})

            start_date = datetime(year=2004,month=2,day=11)#記錄上個月的第一個開盤日
            one_month_day = 0#用來記算每個月的天數
            new_point = 0#記錄新月分第一個開盤日的收盤價
            pre_point = 0#記錄上一個月第一個開盤日的收盤價
            pre_day = 0#記錄前一個迴圈日期的日
            max_month_day = 24#一個月最多24個開盤日

            count = 0
            for date in stock_data['Date']:
                count += 1
                #若上一個回圈的日比這個回圈的日大代表已經到下一個月了,且date為新月份第一個日期
                if date.day<pre_day:
                    print('-')
                    #把上一個月份的日期區間資料append到full_x
                    full_x = np.concatenate((full_x,stock_data[stock_data['Date']<date][stock_data['Date']>=start_date][all_col].values),axis=0)
                    #用這個月最後一開盤日數據補足剩餘不滿24天的數據
                    for null_date in range(max_month_day-one_month_day):

                                                   #! full_x 是array,   後面這邊是dataframe, 這樣np.append沒問題嗎(?)
                        print(full_x.shape) #! 這個full_x的維度只有一維? 為什麼不是48個col所以有48維?
                        full_x = np.concatenate((full_x,stock_data[stock_data['Date'] == full_x[-1][-1]][all_col]),axis=0)
                        print(full_x.shape)

                    #新月份第一個開盤日的收盤價
                    new_point = stock_data[stock_data['Date'] == date]['close'].values
                    #把新月份減舊月份收盤價放到舊月份第一個開盤日
                    if new_point-pre_point>0:
                        full_y = full_y.append({'y':1,'Date':start_date},ignore_index=True)
                    elif new_point-pre_point==0:
                        full_y = full_y.append({'y':0,'Date':start_date},ignore_index=True)
                    else:
                        full_y = full_y.append({'y':-1,'Date':start_date},ignore_index=True)
                    #更新start_date
                    start_date = date

                    pre_point = new_point
                    one_month_day = 1
                    pre_day = date.day
                else:
                    one_month_day = one_month_day + 1
                pre_day = date.day
                if count == 20:
                    exit()

                print(pre_day)
                print('-'*30)

    print(full_x.shape)
    print(full_y.shape)
    

    return full_x, full_y



if '__main__' == __name__:

    files = os.listdir('./month_predict_input')
    all_col = pickle.load(open('selected_cols_for_total_data.pkl', 'rb'))
    all_col.append('Date')
    all_col = [s for s in all_col if s != 'y_updown' and s != 'y_percent' and s != 'y_point']

    full_x = np.empty((0,len(all_col)))
    #full_y = np.empty((0,1))
    full_y = pd.DataFrame(columns = ['y','Date'])

    full_x,full_y = combine_files(files,all_col,full_x,full_y)
    full_x = pd.DataFrame(full_x,columns = all_col)

    pickle.dump(full_x,open(f'./monthfin_lstm_x_input.pkl','wb'))
    pickle.dump(full_y,open(f'./monthfin_lstm_y_input.pkl','wb'))

