import numpy as np
from datetime import datetime
import pandas as pd
import pickle
import torch
import tensorflow as tf
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
#from keras import backend as K
#from tensorflow import keras
from tensorflow.keras import backend as K
from numba import jit
from sklearn.metrics import classification_report

#tf.compat.v1.disable_eager_execution()

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



class Time2Vector(tf.keras.layers.Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)
    def call(self, x):
        x = tf.math.reduce_mean(x[:,:,:100], axis=-1) 
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1) 
        return tf.concat([time_linear, time_periodic], axis=-1) 

class SingleAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = tf.keras.layers.Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.key = tf.keras.layers.Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.value = tf.keras.layers.Dense(self.d_v, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs): 
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out  


class MultiAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
          self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
        self.linear = tf.keras.layers.Dense(131, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear 

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):

        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.attn_normalize = tf.keras.layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        #self.ff_conv1D_1 = tf.keras.layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        #self.ff_conv1D_2 = tf.keras.layers.Conv1D(filters=131, kernel_size=1) # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
        
        self.ff_Dense_1 = tf.keras.layers.Dense(self.ff_dim, activation='relu')
        self.ff_Dense_2 = tf.keras.layers.Dense(131, activation='relu')
        self.ff_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.ff_normalize = tf.keras.layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)    

    def call(self, inputs): 

        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        #ff_layer = self.ff_conv1D_1(attn_layer)
        #ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_Dense_1(attn_layer)
        ff_layer = self.ff_Dense_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer 


def create_model():

    time_embedding = Time2Vector(24)
    in_seq = tf.keras.Input(shape=(train_x.shape[1], train_x.shape[2]),batch_size=17)
    x = time_embedding(in_seq)
    attn_layer1 = TransformerEncoder(64, 64, 2, 64)
    attn_layer2 = TransformerEncoder(32, 32, 2, 32)

    x = tf.keras.layers.Concatenate(axis=-1)([in_seq, x])
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=in_seq, outputs=out)
    return model

@jit
def train(model, train_x, train_y):

    print(model.summary())
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])#,f1_m,precision_m, recall_m])

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_x, tf.keras.utils.to_categorical(train_y,  num_classes=3), epochs=100, batch_size=17, callbacks=[tensorboard_callback])
    return model



if '__main__' == __name__:

    result = pickle.load(open('./store_pkl/transformers_result.pkl','rb'))
    print(result)
    exit()

    data_x = pickle.load(open('./store_pkl/add_lstm_x_input.pkl','rb'))
    data_y = pickle.load(open('./store_pkl/new_y.pkl','rb'))
    data_y = data_y.set_index('Date')

    div = datetime(2018, month=12, day=31)
    train_x = data_x[data_x['Date']<=div]
    test_x = data_x[data_x['Date']>div]
    train_x = data_x[data_x['Date']<=div]
    test_x = data_x[data_x['Date']>div]

    train_y = data_y[data_y.index<=div]
    train_y = train_y.values
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
    
    model = create_model()
    model = train(model, train_x, train_y)

    #loss, accuracy, f1_score, precision, recall
    loss, accuracy = model.evaluate(test_x, tf.keras.utils.to_categorical(test_y, num_classes=3), verbose=1)

    testPredict = model.predict(test_x, verbose=1)
    #pickle.dump(testPredict,open('./transformers_testPredict.pkl','wb'))
    y_result = np.argmax(testPredict,axis=1)
    test_y = test_y[:,0]

    y_result[y_result==2] = -1
    test_y[test_y==2] = -1
    test_y = test_y.astype(np.int)

    report = classification_report(test_y, y_result)
    print(report)

    exit()
    pickle.dump(report,open('./transformers_result.txt','wb'))

    
    div = datetime(2018, month=12, day=31)
    testPredict = pickle.load(open('./transformers_testPredict.pkl','rb'))
    y_result = np.argmax(testPredict,axis=1)
    y_result[y_result==2] = -1
    profit = sharpe_ratio(data_x[data_x['Date']>div],y_result)
    pickle.dump(profit,open('./transformers_profit.pkl','wb'))

   











