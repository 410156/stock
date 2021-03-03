import numpy as np
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import pickle

def page_num():

    typed_table={ 'EB011000E':'水泥',
                  'EB012000E':'食品',
                  'EB013000E':'塑膠',
                  'EB014000E':'紡織纖維',
                  'EB015000E':'電機機械',
                  'EB016000E':'電器電纜',
                  'EB030000E':'化工',
                  'EB031000E':'生技醫療',
                  'EB018000E':'玻璃陶瓷',
                  'EB019000E':'造紙',
                  'EB020000E':'鋼鐵',
                  'EB021000E':'橡膠',
                  'EB022000E':'汽車',
                  'EB032000E':'半導體',
                  'EB033000E':'電腦及週邊設備',
                  'EB034000E':'光電',
                  'EB035000E':'通信網路',
                  'EB036000E':'電子零組件',
                  'EB037000E':'電子通路',
                  'EB038000E':'資訊服務',
                  'EB039000E':'其他電子',
                  'EB025000E':'建材營造',
                  'EB026000E':'航運業',
                  'EB027000E':'觀光',
                  'EB028000E':'金融保險',
                  'EB029000E':'貿易公司',
                  'EB040000E':'油電燃氣',
                  'EB091000E':'存託憑證',
                  'EB099000E':'其他',
                  'EB000000E':'基金',
                  'EB09990XE':'展延型牛熊證',
                  'EB09990RE':'特別股公司債'
                 }

    return typed_table



def stock_table_crawler(typed_table):

    stock_table={}

    for page_num in typed_table:

       # page = pd.read_html("https://fubon-ebrokerdj.fbs.com.tw/z/ze/zeg/zeg_%s_I.djhtm"%(page_num),encoding='big5hkscs')
        page = pd.read_html(f'https://fubon-ebrokerdj.fbs.com.tw/z/ze/zeg/zeg_{page_num}_I.djhtm', encoding='big5hkscs')
        typed = typed_table[page_num]
        stock_table[typed] = {}

        for i in range(2,len(page[2])):
            name_num=re.findall('\'*.....\', \'.*\'',str(page[2].iloc[i][0]))[0]
            name=name_num.split(',' , 1)[1].strip("'\'").strip(" '\'")
            
            stock_table[typed][name]={}
            stock_table[typed][name]['code']=name_num.split(',' , 1)[0].strip("'\'")
            columns = ['close', 'increase_decrease', 'increase_decrease_rate', 'open', 'high', 'low', 'number']

            for x, y in enumerate(columns):
                stock_table[typed][name][y] = str(page[2].iloc[i][x+1])

    return stock_table




if '__main__' == __name__ :

    typed_table= page_num()

    stock_table = stock_table_crawler(typed_table)
    pickle.dump(stock_table, open('./total_data/total.pkl', 'wb'))
       
    print(stock_table['水泥']['台泥'])
