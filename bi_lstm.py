import math
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time
from sklearn.model_selection import GridSearchCV
from keras.metrics import categorical_accuracy
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
import os


def sharpe_ratio(test_x,predict_result):

   
    month_first_price = [test_x['close'].values[i] for i in range(0,len(test_x['close']),24)]
    stock_code = [test_x['stock_num'].values[i] for i in range(0,len(test_x['stock_num']),24)]
    time = [pd.to_datetime(str(test_x['Date'].values[i])) for i in range(0,len(test_x['Date']),24)]
    
    profit = {}
    for stock_num in stock_code:
        profit[stock_num] = {}

    for predict in range(len(predict_result)-1):

        time_period = time[predict].strftime("%Y/%m/%d") + '-' + time[predict+1].strftime("%Y/%m/%d")
        if predict_result[predict] == 1:
            profit[stock_code[predict]][time_period] = (month_first_price[predict+1]-month_first_price[predict])/month_first_price[predict]              

        elif predict_result[predict] == -1:
            profit[stock_code[predict]][time_period] = (month_first_price[predict]-month_first_price[predict+1])/month_first_price[predict]              

    return profit

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if '__main__' == __name__:

    
    #tf.compat.v1.disable_eager_execution()
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.7
    #session = tf.compat.v1.Session(config=config)

    '''
    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    assert len(gpu) == 1
    tf.config.experimental.set_memory_growth(gpu[0],True)

    '''


    #data_x = pickle.load(open('./new_ind_lstm_x_input.pkl','rb'))
    #data_y = pickle.load(open('./new_ind_lstm_y_input.pkl','rb'))
    #select_feature = pickle.load(open('./add_xgb_feature_select.pkl','rb'))

     
    data_x = pickle.load(open('./add_lstm_x_input.pkl','rb'))    
    #data_x = data_x.loc[:,select_feature[50]['feature_names']]

    
    data_y = pickle.load(open('./add_lstm_y_input.pkl','rb'))

    data_y = data_y.set_index('Date')

    div = datetime(2018, month=12, day=31)
    train_x = data_x[data_x['Date']<=div]
    test_x = data_x[data_x['Date']>div]
    train_y = data_y[data_y.index<=div].values
    test_y = data_y[data_y.index>div].values
   
    for f in train_x.columns:
        train_x[f] = pd.to_numeric(train_x[f])
        test_x[f] = pd.to_numeric(test_x[f])
    
    train_x = train_x.fillna(0)
    test_x = test_x.fillna(0)


    labelencoder = LabelEncoder()
    train_x['Date'] = labelencoder.fit_transform(train_x['Date'])
    train_x['stock_num'] = labelencoder.fit_transform(train_x['stock_num'])
    test_x['Date'] = labelencoder.fit_transform(test_x['Date'])
    test_x['stock_num'] = labelencoder.fit_transform(test_x['stock_num'])


    train_y[train_y==-1] = 2
    test_y[test_y==-1] = 2

    sc = MinMaxScaler(feature_range = (0, 1))
    training_x_set_scaled = sc.fit_transform(train_x)
    testing_x_set_scaled = sc.fit_transform(test_x)

    train_x = np.reshape(training_x_set_scaled, ((int)(training_x_set_scaled.shape[0]/24), 24, training_x_set_scaled.shape[1]))
    test_x = np.reshape(testing_x_set_scaled, ((int)(testing_x_set_scaled.shape[0]/24), 24, testing_x_set_scaled.shape[1]))

    model = Sequential()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)

    forward_layer = LSTM(128, return_sequences=True)
    backward_layer = LSTM(128, activation='relu', return_sequences=True,go_backwards=True)
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer,input_shape=(train_x.shape[1], train_x.shape[2])))
    activation = model.add(Bidirectional(LSTM(64)))
    '''
    attention = Dense(1, activation='tanh')(activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)
    attention = Permute([2, 1])(attention)
    '''



    model.add(Dense(32,activation="relu",name="FC1"))
    model.add(Dropout(0.2))

    model.add(Dense(3, activation='softmax'))
    model.summary()

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',f1_m,precision_m, recall_m])
    model.fit(train_x, tf.keras.utils.to_categorical(train_y,  num_classes=3), epochs=100, batch_size=10,callbacks=[callback])
    loss, accuracy, f1_score, precision, recall = model.evaluate(test_x, tf.keras.utils.to_categorical(test_y, num_classes=3), verbose=1)

    testPredict = model.predict(test_x, verbose=1)
    pickle.dump(testPredict,open('./bilstm_testPredict.pkl','wb'))
    y_result = np.argmax(testPredict,axis=1)
    test_y = test_y[:,0]

    y_result[y_result==2] = -1
    test_y[test_y==2] = -1
    test_y = test_y.astype(np.int)


    

    report = classification_report(test_y, y_result,output_dict=True)
    print(report)
    report1 = classification_report(test_y, y_result)
    print(report1)

    pickle.dump(report1,open('./bilstm_result.pkl','wb'))

    profit = sharpe_ratio(data_x[data_x['Date']>div],y_result)
    print(profit)
    pickle.dump(profit,open('./bilstm_profit.pkl','wb'))
    #exit() 

    with open('.bilstm_report.json','w') as outfile:
        json.dump(report,outfile)
    f = open('./bilstm_result.txt', 'w')
    f.write(report1)
    f.close()
    

