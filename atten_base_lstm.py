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
#import tensorflow as tf
import torch
from torch import nn, optim
import torch.nn.functional as F


LEARNING_RATE = 1e-3 

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

def binary_acc(preds, y):
     preds = torch.round(torch.sigmoid(preds))
     correct = torch.eq(preds, y).float()
     acc = correct.sum() / len(correct)
     return acc    
 
 
def train(rnn, train_x, optimizer, criteon, train_y):
 
    avg_loss = []
    avg_acc = []
    rnn.train()    


    for data in range(0,train_x.shape[0],8):
        

        x = train_x[data:data+8,:,:]
        x = x.reshape(x.shape[1]*x.shape[2] ,x.shape[0])
        batch = torch.tensor(x).to(torch.long)
        #print(batch)
     
        pred_loss, pred_acc = rnn(batch)
        pred_loss = pred_loss.squeeze()             
        pred_acc = pred_acc.squeeze()             
         
        y = train_y[data:data+8,:].astype(int)
        y = torch.tensor(y)
        y = y.reshape(y.shape[0]*y.shape[1])
        #print(pred)
        #print(y)
        

        loss = criteon(pred_loss, y)
        #print(loss)
        acc = binary_acc(pred_acc,  y).item()   
         
        avg_loss.append(loss.item())
        avg_acc.append(acc)
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         
    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc                          

 
def evaluate(rnn, test_x, criteon, test_y):    
 
    avg_loss = []
    avg_acc = []    
    rnn.eval()     

    with torch.no_grad():
     for batch in range(0,test_x.shape[0],17):

         x = test_x[batch:batch+17,:,:]
         x = x.reshape(x.shape[1]*x.shape[2] ,x.shape[0])
         batch = torch.tensor(x).to(torch.long)
         
         pred_loss, pred_ac = rnn(batch)
         pred_loss = pred_loss.squeeze()        
         pred_acc = pred_acc.squeeze()        
         
         y = test_y[batch:batch+17,:].astype(float)
         y = torch.tensor(y)
         y = y.reshape(y.shape[0]*y.shape[1])

         loss = criteon(pred_loss, y)
         acc = binary_acc(pred_acc, y).item()
         
         avg_loss.append(loss.item())
         avg_acc.append(acc)
     
    avg_loss = np.array(avg_loss).mean()
    avg_acc = np.array(avg_acc).mean()
    return avg_loss, avg_acc

class BiLSTM_Attention(nn.Module):  
     
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
     
        super(BiLSTM_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)      
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        #nn.init.kaiming_uniform_(self.w_omega, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.u_omega, mode='fan_in', nonlinearity='relu')

    def attention_net(self, x):       
     
        u = torch.tanh(torch.matmul(x, self.w_omega))         
        att = torch.matmul(u, self.u_omega)                 
        att_score = F.softmax(att, dim=1) 
              
        scored_x = x * att_score                              

        context = torch.sum(scored_x, dim=1)                  
        return context


    def forward(self, x):     
        embedding = self.dropout(self.embedding(x))       

        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)                  
         
        attn_output = self.attention_net(output)
        logit = self.fc(attn_output)
        return attn_output,logit







if '__main__' == __name__:


    #data_x = pickle.load(open('./new_ind_lstm_x_input.pkl','rb'))
    #data_y = pickle.load(open('./new_ind_lstm_y_input.pkl','rb'))
    #select_feature = pickle.load(open('./add_xgb_feature_select.pkl','rb'))

    data_x = pickle.load(open('./store_pkl/add_lstm_x_input.pkl','rb'))    
    #data_x = data_x.loc[:,select_feature[50]['feature_names']]

    
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

    rnn = BiLSTM_Attention(train_x.shape[0], train_x.shape[1], hidden_dim=64, n_layers=2)

    optimizer = optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
    #criteon = nn.BCEWithLogitsLoss()
    criteon = nn.CrossEntropyLoss()


    sc = MinMaxScaler(feature_range = (0, 1))
    training_x_set_scaled = sc.fit_transform(train_x)
    testing_x_set_scaled = sc.fit_transform(test_x)

    train_x = np.reshape(training_x_set_scaled, ((int)(training_x_set_scaled.shape[0]/24), 24, training_x_set_scaled.shape[1]))
    test_x = np.reshape(testing_x_set_scaled, ((int)(testing_x_set_scaled.shape[0]/24), 24, testing_x_set_scaled.shape[1]))

    '''
    best_valid_acc = float('-inf')    

    for epoch in range(30):
     
        start_time = time.time()

        train_loss, train_acc = train(rnn, train_x, optimizer, criteon, train_y)

        end_time = time.time()

        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(train_acc)

        if train_acc > best_valid_acc:       
            best_valid_acc = train_acc
            torch.save(rnn.state_dict(), 'wordavg-model.pt')
         
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        #print(f'\t Val. Loss: {dev_loss:.3f} |  Val. Acc: {dev_acc*100:.2f}%')

    '''

    rnn.load_state_dict(torch.load("wordavg-model.pt"))   
    
    test_loss, test_acc = evaluate(rnn, test_x, criteon, test_y)
    print(f'Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')
        
    exit()

    testPredict = model.predict(test_x, verbose=1)
    pickle.dump(testPredict,open('./atten_lstm_testPredict.pkl','wb'))
    y_result = np.argmax(testPredict,axis=1)
    test_y = test_y[:,0]

    y_result[y_result==2] = -1
    test_y[test_y==2] = -1
    test_y = test_y.astype(np.int)


    

    report = classification_report(test_y, y_result,output_dict=True)
    print(report)
    report1 = classification_report(test_y, y_result)
    print(report1)

    pickle.dump(report1,open('./atten_lstm_result.pkl','wb'))

    profit = sharpe_ratio(data_x[data_x['Date']>div],y_result)
    print(profit)
    pickle.dump(profit,open('./atten_lstm_profit.pkl','wb'))
    #exit() 

    with open('.atten_lstm_report.json','w') as outfile:
        json.dump(report,outfile)
    f = open('./atten_lstm_result.txt', 'w')
    f.write(report1)
    f.close()
    

