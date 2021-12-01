import math
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time
from sklearn.model_selection import GridSearchCV
#from keras.metrics import categorical_accuracy
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D, BatchNormalization,Activation
from tensorflow.keras import regularizers


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

    result = pickle.load(open('./store_pkl/cnn_result.pkl','rb'))    
    print(result)
    exit()

    data_x = pickle.load(open('./store_pkl/add_lstm_x_input.pkl','rb'))    
    data_y = pickle.load(open('./store_pkl/add_lstm_y_input.pkl','rb'))
    data_y = data_y.set_index('Date')

    div = datetime(2018, month=12, day=31)
    train_x = data_x[data_x['Date']<=div]
    test_x = data_x[data_x['Date']>div]
    train_y = data_y[data_y.index<=div].values
    test_y = data_y[data_y.index>div].values

    profit_attention = (train_x['close'] - train_x['close'].shift(1,axis=0)) / train_x['close'].shift(1,axis=0)
   
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

    train_x = np.reshape(training_x_set_scaled, ((int)(training_x_set_scaled.shape[0]/24), 24, training_x_set_scaled.shape[1],1))
    test_x = np.reshape(testing_x_set_scaled, ((int)(testing_x_set_scaled.shape[0]/24), 24, testing_x_set_scaled.shape[1],1))

    model = Sequential()
    #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)

    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(train_x.shape[1], train_x.shape[2],1),kernel_initializer='he_uniform',bias_initializer='RandomNormal', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',kernel_initializer='he_uniform',bias_initializer='RandomNormal', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same',kernel_initializer='he_uniform',bias_initializer='RandomNormal', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same',kernel_initializer='he_uniform',bias_initializer='RandomNormal', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same',kernel_initializer='he_uniform',bias_initializer='RandomNormal', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_initializer='he_uniform',bias_initializer='RandomNormal'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu',kernel_initializer='he_uniform',bias_initializer='RandomNormal'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax',kernel_initializer='he_uniform',bias_initializer='RandomNormal')) 
    model.summary()

    #solve_cudnn_error()

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',f1_m,precision_m, recall_m])
    model.fit(train_x, tf.keras.utils.to_categorical(train_y,  num_classes=3), epochs=100, batch_size=10)#,callbacks=[callback])
    loss, accuracy, f1_score, precision, recall = model.evaluate(test_x, tf.keras.utils.to_categorical(test_y, num_classes=3), verbose=1)

    testPredict = model.predict(test_x, verbose=1)
    pickle.dump(testPredict,open('./cnn_testpredict.pkl','wb'))
    y_result = np.argmax(testPredict,axis=1)
    test_y = test_y[:,0]

    y_result[y_result==2] = -1
    test_y[test_y==2] = -1
    test_y = test_y.astype(np.int)

    report = classification_report(test_y, y_result)
    print(report)
    pickle.dump(report,open('./cnn_result.pkl','wb'))

    profit = sharpe_ratio(data_x[data_x['Date']>div],y_result)
    pickle.dump(profit,open('./cnn_profit.pkl','wb'))

