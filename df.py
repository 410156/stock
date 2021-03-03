import numpy as np
import pandas as pd
import json
import os
import datetime
import sys
import pickle

def load_data(ten_stocks, _file, path):

    total_data = {}
    for z in _file:
        with open(path+z, 'rb') as f:
            data = json.load(f)
            for stock in ten_stocks:
                if stock not in total_data.keys():
                    total_data[stock] = []
                if stock not in data.keys():
                    total_data[stock].append([datetime.datetime.strptime(z.strip('.json'), '%Y-%m-%d'), stock, np.nan, np.nan, np.nan, np.nan])
                    
                else: 
                    total_data[stock].append([datetime.datetime.strptime(z.strip('.json'), '%Y-%m-%d'), stock, float(data[stock]['open']), float(data[stock]['high']), float(data[stock]['low']), float(data[stock]['close'])])
    return total_data




if '__main__' == __name__:

    path = '/home/db/stock_resource_center/resource/twse/json/'
    _file = sorted(os.listdir(path))
    ten_stocks=['8112']#, '2324', '5443', '1210', '1720', '3501', '2404', '6470', '2731', '1530']
    #ten_stocks = ['2330', '2303', '0050', '2891']
    
    total_data = load_data(ten_stocks, _file, path)

    for code in total_data.keys():
        total_data[code] = np.array(total_data[code])

    to_df = {}
    for key in total_data.keys():
        to_df[key] = pd.DataFrame(total_data[key], columns=['Date', 'stock_num', 'Open', 'High', 'Low', 'Close'])
        pickle.dump(to_df[key], open(f'./data/{key}.pkl', 'wb'))
print(to_df)
