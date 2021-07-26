import numpy as np
import pandas as pd
import pickle
from pandas import Series
import os
from datetime import datetime

if '__main__' == __name__:

    '''
    #count the max number of day in a month
    merge_y = pickle.load(open('./new_monthfin_lstm_x_input.pkl', 'rb'))
    merge_y = merge_y.reshape((84048,48))
    merge_y = pd.DataFrame(merge_y)
    print(merge_y.iloc[:,-1:])
    exit()
    count=0
    tmp = 1
    day = 0
    for a in merge_x['Date']:
        if a.day<day:
            if tmp>count:
                count = tmp
            tmp=1
        else:
            tmp = tmp+1
        day = a.day
    print(count)
    exit()
    '''
    files_with = os.listdir('./month_predict_input')
    all_ = pickle.load(open('selected_cols_for_total_data.pkl', 'rb'))
    all_.append('Date')
    all_col = [s for s in all_ if s != 'y_updown' and s != 'y_percent' and s != 'y_point']

    full_x_array = np.empty((0,len(all_col)))
    full_y_array = np.empty((0,1))

    for x in range(len(files_with)):
        with open(f'./month_predict_input/{files_with[x]}' , 'rb') as f:
            stock_code = files_with[x].replace('_month_fin_merge.pkl', '')
            print(stock_code)
            predict_content = pickle.load(f)
            predict_content = predict_content.reset_index().rename(columns={'index': 'Date'})

            pre_day = 0
            start_date = datetime(year=2004,month=2,day=11)
            one_month_day = 0
            max_month_day = 24
            feature = predict_content.values.shape[1]
            pre_point = 0
            new_point = 0
            count_day = 0
            for date in predict_content['Date']:
                if date.day<pre_day:
                    full_x_array = np.append(full_x_array,predict_content[predict_content['Date']<date][predict_content['Date']>=start_date][all_col].values)
                    for null_date in range(max_month_day-one_month_day):
                        full_x_array = np.append(full_x_array,predict_content[predict_content['Date'] == full_x_array[-1]][all_col])
                    start_date = date
                    pre_day = count_day
                    count_day = count_day + one_month_day 
                    new_point = predict_content['close'][count_day]
                    y_date = predict_content['Date'][pre_day]
                    if new_point-pre_point>0:
                        full_y_array = np.append(full_y_array,1)
                    elif new_point-pre_point==0:
                        full_y_array = np.append(full_y_array,0)
                    else:
                        full_y_array = np.append(full_y_array,-1)
                    full_y_array = np.append(full_y_array,y_date)
                    pre_point = new_point
                    one_month_day = 1
                    pre_day = date.day
                    
                else:
                    one_month_day = one_month_day + 1
                pre_day = date.day
    print(full_x_array)
    print(full_y_array)
    pickle.dump(full_x_array,open(f'./new_monthfin_lstm_x_input.pkl','wb'))
    pickle.dump(full_y_array,open(f'./new_monthfin_lstm_y_input.pkl','wb'))

