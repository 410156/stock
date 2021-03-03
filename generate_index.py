import numpy as np
import pandas as pd
import json
import os
import datetime
import sys
import pickle
import requests


def kd(target):
    
    target['K'], target['D'], target['RSV'] = 0, 0, 0
    target['RSV'] = 100*(target['Close']-target['Low'].rolling(9).min())/(target['High'].rolling(9).max()-target['Low'].rolling(9).min())
    target['K']= 2*target['K'].shift(periods=1)/3 + target['RSV']
    target['D']= target['D'].shift(periods=1)*2/3 + target['K']/3 
    
    return target

def ema(target):
    target['EMA(12)'], target['EMA(26)'], target['DIF'] =  target['Close']*2/13, target['Close']*2/27, target['Close']*2/13-target['Close']*2/27
    target['EMA(12)']=(target['EMA(12)'].shift(periods=1)*11+2*target['Close'])/13
    target['EMA(26)']=(target['EMA(26)'].shift(periods=1)*25+2*target['Close'])/27
    target['DIF']=target['EMA(12)']-target['EMA(26)']
    return target

def macd(target, x=9):
    #for key in target.keys():
    #    start=np.where(pd.isnull(target['Close'].values)==False)[0][0]
    #if(==start): 
    target['MACD']=target['DIF']*2/(x+1)
    #elif(target>start): 
    target['MACD']=(target['MACD'].shift(periods=1)*(x-1)+target['DIF']*2)/(x+1)
    return target

if '__main__' == __name__:
    target = pickle.load(open('./data/8112.pkl', 'rb'))
    target = kd(target)
    target=ema(target)
    target=macd(target, x=9)
    pickle.dump(target, open('./include_KDdata/8112.pkl', 'wb'))
    print(target)
    
