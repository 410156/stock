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
import tensorflow as tf
import torch
from torch import nn, optim
import torch.nn.functional as F
from numba import jit
from keras import backend as K
from torch.autograd import Variable
import matplotlib.pyplot as plt



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


def binary_acc(preds, y):
     preds = torch.round(torch.sigmoid(preds))
     correct = torch.eq(preds, y).float()
     acc = correct.sum() / len(correct)
     return acc    

def f1_score(preds, y):

    tp_dec = 0
    fp_dec = 0
    fn_dec = 0

    for i in range(len(preds)):
        if preds[i] == 2:            
            if y[i] == 2:
                tp_dec = tp_dec + 1
            else:
                fp_dec = fp_dec + 1
        else:
            if y[i] == 2:
                fn_dec = fn_dec + 1

    if tp_dec + fp_dec == 0:
        precision_dec = 0
    else:
        precision_dec = tp_dec / ( tp_dec + fp_dec )


    if tp_dec + fn_dec == 0:
        recall_dec = 0
    else:
        recall_dec = tp_dec / ( tp_dec + fn_dec )

    if precision_dec == 0 or recall_dec == 0:
        f1_dec = 0
    else:
        f1_dec = 2 / ( ( 1 / precision_dec ) + ( 1 / recall_dec ) )

    tp_inc = 0
    fp_inc = 0
    fn_inc = 0

    for i in range(len(preds)):
        if preds[i] == 1:            
            if y[i] == 1:
                tp_inc = tp_inc + 1
            else:
                fp_inc = fp_inc + 1
        else:
            if y[i] == 1:
                fn_inc = fn_inc + 1

    if tp_inc + fp_inc == 0:
        precision_inc = 0
    else :
        precision_inc = tp_inc / ( tp_inc + fp_inc )


    if tp_inc + fn_inc == 0:
        recall_inc = 0
    else:
        recall_inc = tp_inc / ( tp_inc + fn_inc )


    if precision_inc == 0 or recall_inc == 0:
        f1_inc = 0
    else:
        f1_inc = 2 / ( ( 1 / precision_inc ) + ( 1 / recall_inc ) )

    f1_avg = ( f1_dec + f1_inc ) / 2

    return torch.tensor(f1_avg)


 
def profit_loss(pred, y, close, record_loss):

    loss = record_loss    
    close = np.array(close)
    
    close[np.isnan(close)] = 0

    for i in range(len(close)-1):

        profit = abs((close[i+1] - close[i]) / close[i])

        if close[i] == 0:
            profit = 0
        

        if pred[i] == y[i]:
            loss = loss - profit
        else:
            loss = loss + profit

    loss = torch.tensor(loss).to(torch.float32)
    return loss


 
@jit
def train(rnn, train_x, optimizer, criteon, train_y):
#def train(rnn, train_x, optimizer, train_y, close):
 
    avg_loss = []
    avg_acc = []
    rnn.train()    


    #month_first_price = [close[i] for i in range(0,len(close),24)]
    #record_loss = 0
    
    for data in range(0,train_x.shape[0],8):
        
        x = train_x[data:data+8,:,:]
        #x = x.reshape(x.shape[1]*x.shape[2] ,x.shape[0])
        #batch = torch.tensor(x).to(torch.long)
        batch = torch.tensor(x).to(torch.float)
        
     
        pred_loss, pred_acc = rnn(batch)
        pred_loss = pred_loss.squeeze()             
        pred_acc = pred_acc.squeeze()             
         
        y = train_y[data:data+8,:].astype(int)
        y = torch.tensor(y)
        y = y.reshape(y.shape[0]*y.shape[1])

        #one_hot_target = torch.zeros(y.shape[0], 3)
        #one_hot_target[torch.arange(y.shape[0]), y] = 1
        

        #loss = criteon(torch.sigmoid(pred_loss), one_hot_target)
        #print(pred_acc)
        #print(y)
        loss = criteon(pred_loss, y)

        #loss = profit_loss(pred_acc, y, month_first_price[data:data+9],record_loss)
        #record_loss = loss
        acc = f1_score(pred_acc, y)

        avg_loss.append(loss.item())
        avg_acc.append(acc)
         
        optimizer.zero_grad()
        loss = Variable(loss, requires_grad=True)
        loss.backward()
        optimizer.step()
         
    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc                          

 
#@jit
def evaluate(rnn, test_x, criteon, test_y):    
#def evaluate(rnn, test_x, test_y, close):    
 
    avg_loss = []
    avg_acc = []    
    rnn.eval()     
    #month_first_price = [close[i] for i in range(0,len(close),24)]
    #record_loss = 0
    y_pred = torch.empty(0)
    y_true = torch.empty(0)

    with torch.no_grad():
        for data in range(0,test_x.shape[0],8):


            x = test_x[data:data+8,:,:]
            #x = x.reshape(x.shape[1]*x.shape[2] ,x.shape[0])
            #batch = torch.tensor(x).to(torch.long)
            batch = torch.tensor(x).to(torch.float)

            pred_loss, pred_acc = rnn(batch)
            pred_loss = pred_loss.squeeze()        
            pred_acc = pred_acc.squeeze()        

            y = test_y[data:data+8].astype(int)
            y = torch.tensor(y)
            y = y.reshape(y.shape[0]*y.shape[1])

            y_pred = torch.cat((y_pred,pred_acc))
            y_true = torch.cat((y_true,y))


            one_hot_target = torch.zeros(y.shape[0], 3)
            one_hot_target[torch.arange(y.shape[0]), y] = 1
        
            #loss = criteon(torch.sigmoid(pred_loss), one_hot_target)
            
            loss = criteon(pred_loss, y)
            #loss = profit_loss(pred_acc, y, month_first_price[data:data+9],record_loss)
            #record_loss = loss

            avg_loss.append(loss.item())
            #avg_acc.append(acc)


    avg_loss = np.array(avg_loss).mean()
    #avg_acc = np.array(avg_acc).mean()
    avg_acc = classification_report(y_true, y_pred)

    return avg_loss, avg_acc, y_pred
'''
class BiLSTM_Attention(nn.Module):  
     
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
     
        super(BiLSTM_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)      
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 2, 3)

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
        attn_output = self.fc(attn_output)
        logit = torch.argmax(attn_output, dim=1)

        return attn_output,logit
'''

n_hidden = 5

class BiLSTM_Attention(nn.Module):
    #def __init__(self, vocab_size, embedding_dim, num_classes, n_hidden):
    def __init__(self, embedding_dim,num_classes):
        super(BiLSTM_Attention, self).__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    
    def attention_net(self,lstm_output, final_state):
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        batch_size = len(lstm_output)
        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0],final_state[1]),dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights : [batch_size,n_step]
        soft_attn_weights = F.softmax(attn_weights,1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1,2),soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context, soft_attn_weights
    
    def forward(self,X):
        '''
        input = self.embedding(X) # input : [batch_size, seq_len, embedding_dim]
        #input = input.transpose(0, 1) # input : [seq_len, batch_size, embedding_dim]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # output : [seq_len, batch_size, n_hidden * num_directions(=2)]
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        #output = output.transpose(0, 1) #output : [batch_size, seq_len, n_hidden * num_directions(=2)]
        
        output = output[-1]  # [batch_size, n_hidden * 2]
        '''
        batch_size = X.shape[0]
        input = X.transpose(0, 1)  # input : [max_len, batch_size, n_class]

        hidden_state = torch.randn(1*2, batch_size, n_hidden) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1*2, batch_size, n_hidden) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        output, (_, _) = self.lstm(input, (hidden_state, cell_state))
        output = output.transpose(0, 1) #output : [batch_size, seq_len, n_hidden * num_directions(=2)]
        #output = output[-1]  # [batch_size, n_hidden * 2]

        attn_output, attention = self.attention_net(output,hidden_state)
        return self.out(attn_output),torch.argmax(self.out(attn_output), dim=1) # attn_output : [batch_size, num_classes], attention : [batch_size, n_step]
        #return self.out(output),torch.argmax(self.out(output), dim=1) # attn_output : [batch_size, num_classes], attention : [batch_size, n_step]

        #attn_output, attention = self.attention_net(output,final_hidden_state)
        #return self.out(attn_output),torch.argmax(self.out(attn_output), dim=1) # attn_output : [batch_size, num_classes], attention : [batch_size, n_step]





if '__main__' == __name__:


    #all_col = pickle.load(open('./store_pkl/selected_cols_for_total_data.pkl', 'rb'))
    #all_col.append('Date')
    #all_col = [s for s in all_col if s != 'y_updown' and s != 'y_percent' and s != 'y_point' and s!= 'DIF' and s!='MACD' and s!='K' and s!= 'D' and s!='RSV' and s!='EMA(12)' and s!= 'EMA(26)' and s!='DIF'and s!= '5MA' and s!= '20MA' and s!='60MA' and s!='5MA over 20MA' and s!='20MA over 5MA']
    #all_col = [s for s in all_col if s != 'y_updown' and s != 'y_percent' and s != 'y_point']
    #all_col = ['DIF' ,'MACD' ,'K' ,'D' ,'RSV' ,'EMA(12)' ,'EMA(26)' ,'DIF','5MA','20MA','60MA','5MA over 20MA','20MA over 5MA','open','Date','close' ,'low' ,'high','volume']
    #all_col = [s for s in all_col if s == 'DIF' or s == 'MACD' or s == 'K'  or s =='D'  or s =='RSV'  or s =='EMA(12)'  or s =='EMA(26)'  or s =='DIF' or s =='5MA' or s =='20MA' or s =='60MA' or s =='open' or s =='Date' or s =='close'  or s =='low'  or s =='high' or s =='volume' or s=='stock_num']
    data_x = pickle.load(open('./store_pkl/add_lstm_x_input.pkl','rb'))#.loc[:,all_col]    
    data_y = pickle.load(open('./store_pkl/new_y.pkl','rb'))
    data_y = data_y.set_index('Date')

    div = datetime(2018, month=12, day=31)
    train_x = data_x[data_x['Date']<=div]
    test_x = data_x[data_x['Date']>div]
    train_y = data_y[data_y.index<=div]
    #train_y = train_y[train_y.index>d]
    train_y = train_y.values
    test_y = data_y[data_y.index>div].values

    train_close = train_x['close'].values
    test_close = test_x['close'].values

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

    #rnn = BiLSTM_Attention(train_x.shape[0], train_x.shape[1], hidden_dim=64, n_layers=5)
    #rnn = BiLSTM_Attention(128, 64, hidden_dim=64, n_layers=2)
    #rnn = BiLSTM_Attention(128, 64,3,2)
    rnn = BiLSTM_Attention(train_x.shape[1],3)

    optimizer = optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
    #criteon = nn.BCELoss()
    criteon = nn.CrossEntropyLoss()

    sc = MinMaxScaler(feature_range = (0, 1))
    training_x_set_scaled = sc.fit_transform(train_x)
    testing_x_set_scaled = sc.fit_transform(test_x)

    train_x = np.reshape(training_x_set_scaled, ((int)(training_x_set_scaled.shape[0]/24), 24, training_x_set_scaled.shape[1]))
    test_x = np.reshape(testing_x_set_scaled, ((int)(testing_x_set_scaled.shape[0]/24), 24, testing_x_set_scaled.shape[1]))

    #best_valid_loss = float('-inf')    
    best_valid_loss = 999    
    
    loss = []
    acc = []
    '''
    for epoch in range(250):
     
        start_time = time.time()

        #train_loss, train_acc = train(rnn, train_x, optimizer, criteon, train_y)
        train_loss, train_acc = train(rnn, train_x, optimizer,  train_y, train_close)

        end_time = time.time()

        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        if train_loss < best_valid_loss:       
            best_valid_loss = train_loss
            torch.save(rnn.state_dict(), 'atten_profit_loss_model.pkl')
         
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train avg f1 score: {train_acc*100:.2f}')
       
        loss.append(train_loss) 
        acc.append(train_acc) 
    
    pickle.dump(loss, open('atten_profit_loss_loss.pickle', 'wb'))
    pickle.dump(acc, open('atten_profit_loss_acc.pickle', 'wb'))
    '''
    rnn.load_state_dict(torch.load("atten_bilstm_model.pkl"))   
    
    test_loss, test_acc, y_pred = evaluate(rnn, test_x, criteon, test_y)

    
    pickle.dump(y_pred,open('./atten_bilstm_ypred.pkl','wb'))
    pickle.dump(test_loss,open('./atten_bilstm_pred_loss.pkl','wb'))
    pickle.dump(test_acc,open('./atten_bilstm_pred_f1.pkl','wb'))
    profit = sharpe_ratio(data_x[data_x['Date']>div],y_pred)
    print(profit)
    pickle.dump(profit,open('./atten_bilstm_profit.pkl','wb'))

    #test_loss, test_acc = evaluate(rnn, test_x, test_y, test_close)
    print(f'Test. Loss: {test_loss:.3f}')# |  Test. f1_score: {test_acc*100:.2f}')


    f = open('./atten_bilstm_result.txt', 'w')
    f.write(test_acc)
    f.close()
    
    print(test_acc)
       
