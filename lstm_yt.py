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

    cols = [i for i in cols if i not in ['y_updown', 'y_percent', 'y_point']]

    all_x = np.empty([0, 48])
    all_y = []

    for x in files[:3]:
        with open(f'./month_predict_input/{x}' , 'rb') as f:
            stock_code = x.replace('_month_fin_merge.pkl', '')
            predict_content = pickle.load(f)
            predict_content = predict_content.reset_index().rename(columns={'index': 'Date'})
            dates =  predict_content.Date

            for year in range(2004, 2021):
                for month in range(1, 13):

                    # 找到那個月份的index有哪些後拿出來
                    if month != 12:
                        count = dates.loc[(dates>=datetime(year, month, 1)) & (dates<datetime(year, month+1, 1))].index
                    else:
                        count = dates.loc[(dates>=datetime(year, 12, 1)) & (dates<datetime(year+1, 1, 1))].index
                    wanted = predict_content.iloc[count]


                    # 如果不足24天, 補最後一天的資料補足24天, 順便存y
                    if wanted.shape[0] != 0 and wanted.shape[0] != 24:
                        y = wanted['y_point'].values[-1] - wanted['y_point'].values[0]
                        wanted = wanted[cols].values
                        pad = np.repeat(wanted[-1].reshape(1, -1), 24-wanted.shape[0], axis=0)
                        wanted = np.concatenate((wanted, pad), axis=0)

                        all_x = np.concatenate((all_x, wanted), axis=0)
                        all_y.append(y)

    # 共有606筆資料, 每筆x的維度是24*48
    all_x = all_x.reshape(-1, 24, 48)
    print(all_x.shape)
    print(len(all_y))
