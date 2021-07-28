import numpy as np
import pandas as pd
import pickle
from pandas import Series
import os
from datetime import datetime



def combine_files():






    return all_x_array, all_y_array











if '__main__' == __name__:

    files = os.listdir('./month_predict_input')
    cols = pickle.load(open('./selected_cols_for_total_data.pkl', 'rb'))
    cols.append('Date')
    # cols = [s for s in cols if s not in ['y_updown', 'y_percent', 'y_point']]

    full_x_array = np.empty((0,len(cols)))
    full_y_array = np.empty((0,1))

    for x in files:
        with open(f'./month_predict_input/{x}' , 'rb') as f:
            stock_code = x.replace('_month_fin_merge.pkl', '')
            predict_content = pickle.load(f)
            predict_content = predict_content.reset_index().rename(columns={'index': 'Date'})


            dates =  predict_content.Date

            for year in range(2004, 2021):
                for month in range(1, 13):
                    if month != 12:
                        count = dates.loc[(dates>=datetime(year, month, 1)) & (dates<datetime(year, month+1, 1))].index
                    else:
                        count = dates.loc[(dates>=datetime(year, 12, 1)) & (dates<datetime(year+1, 1, 1))].index
                    wanted = predict_content.iloc[count][cols].values

                    if wanted.shape[0] != 0 and wanted.shape[0] != 24:
                        pad = np.repeat(wanted[-1].reshape(1, -1), 24-wanted.shape[0], axis=0)
                        wanted = np.concatenate((wanted, pad), axis=0)
                        print(wanted.shape)

    #             if date.day<pre_day:
    #                 full_x_array = np.append(full_x_array,predict_content[predict_content['Date']<date][predict_content['Date']>=start_date][all_col].values)
    #                 for null_date in range(max_month_day-one_month_day):
    #                     full_x_array = np.append(full_x_array,predict_content[predict_content['Date'] == full_x_array[-1]][all_col])
    #                 start_date = date
    #                 pre_day = count_day
    #                 count_day = count_day + one_month_day
    #                 new_point = predict_content['close'][count_day]
    #                 y_date = predict_content['Date'][pre_day]
    #                 if new_point-pre_point>0:
    #                     full_y_array = np.append(full_y_array,1)
    #                 elif new_point-pre_point==0:
    #                     full_y_array = np.append(full_y_array,0)
    #                 else:
    #                     full_y_array = np.append(full_y_array,-1)
    #                 full_y_array = np.append(full_y_array,y_date)
    #                 pre_point = new_point
    #                 one_month_day = 1
    #                 pre_day = date.day
    #             else:
    #                 one_month_day = one_month_day + 1
    #             pre_day = date.day
    # print(full_x_array)
    # print(full_y_array)
    # pickle.dump(full_x_array,open(f'./new_monthfin_lstm_x_input.pkl','wb'))
    # pickle.dump(full_y_array,open(f'./new_monthfin_lstm_y_input.pkl','wb'))
    #
