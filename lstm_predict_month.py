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

    data_x = pickle.load(open('./new_monthfin_lstm_x_input.pkl','rb'))
    data_y = pickle.load(open('./new_monthfin_lstm_y_input.pkl','rb'))

    #data_y have y and 'Date'
    data_y = data_y.reshape((3502,2))
    data_y = pd.DataFrame(data_y)
    data_y = data_y.set_index(1)
    #x have 48 feature
    data_x = data_x.reshape((84048,48))
    data_x = pd.DataFrame(data_x)
 
    div = datetime(2018, month=12, day=31)
    train_x = data_x[data_x.iloc[:,47]<=div]
    test_x = data_x[data_x.iloc[:,47]>div]
    train_y = data_y[data_y.index<=div].values
    test_y = data_y[data_y.index>div].values
    
    
    for f in train_x.columns:
        train_x[f] = pd.to_numeric(train_x[f])
        test_x[f] = pd.to_numeric(test_x[f])
    
    train_x = train_x.fillna(0)
    test_x = test_x.fillna(0)

    #col 47:'Date' , col 10:'stock_num'
    labelencoder = LabelEncoder()
    train_x[47] = labelencoder.fit_transform(train_x[47])
    train_x[10] = labelencoder.fit_transform(train_x[10])
    test_x[47] = labelencoder.fit_transform(test_x[47])
    test_x[10] = labelencoder.fit_transform(test_x[10])

    train_y[train_y==-1] = 2
    test_y[test_y==-1] = 2

    sc = MinMaxScaler(feature_range = (0, 1))
    training_x_set_scaled = sc.fit_transform(train_x)
    testing_x_set_scaled = sc.fit_transform(test_x)

    train_x = np.reshape(training_x_set_scaled, ((int)(training_x_set_scaled.shape[0]/24), 24, training_x_set_scaled.shape[1]))
    test_x = np.reshape(testing_x_set_scaled, ((int)(testing_x_set_scaled.shape[0]/24), 24, testing_x_set_scaled.shape[1]))

    model = Sequential()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    model.add(LSTM(128, activation='relu',return_sequences = True, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(32,activation="relu",name="FC1"))
    model.add(Dropout(0.2))

    model.add(Dense(3, activation='softmax'))
    model.summary()

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',f1_m,precision_m, recall_m])
    model.fit(train_x, tf.keras.utils.to_categorical(train_y,  num_classes=3), epochs=100, batch_size=10,callbacks=[callback])
    loss, accuracy, f1_score, precision, recall = model.evaluate(test_x, tf.keras.utils.to_categorical(test_y, num_classes=3), verbose=1)

    testPredict = model.predict(test_x, verbose=1)
    pickle.dump(testPredict,open('./lstm_testPredict.pkl','wb'))
    y_result = np.argmax(testPredict,axis=1)
    test_y = test_y[:,0]

    y_result[y_result==2] = -1
    test_y[test_y==2] = -1
    test_y = test_y.astype(np.int)


    report = classification_report(test_y, y_result,output_dict=True)
    print(report)
    report1 = classification_report(test_y, y_result)
    print(report1)
 
    with open('./lstm_first_report.json','w') as outfile:
        json.dump(report,outfile)
    f = open('./lstm_first_result.txt', 'w')
    f.write(report1)
    f.close()
    exit()
    

