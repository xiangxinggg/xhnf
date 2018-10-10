# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
import numpy as np
import os
import sys
import datetime

# 得到所有股票代码
def get_all_stocks_index():
    df = ts.get_stock_basics()
    all_stock = df.index.tolist()   
    return all_stock

def get_data(code = '000001', min = "5min", start = '20100101', end = '20200101'): #min = "5min"
    #df = ts.bar(code, conn=ts.get_apis(), freq=min, start_date = start, end_date = end)
    #df = ts.get_hist_data(code, start=start, end=end, pause=3)
    df = ts.get_hist_data(code, pause=3)
    return df

def download_data(code, file_name):
    try:
        isExists=os.path.exists(file_name)
        if not isExists:
            print('start download:'+file_name)
            df = get_data(code = stock)
            if df.size == 18:
                print(df)
            else:
                df.to_csv(file_name, encoding="utf_8_sig")
            print('end downloaded:'+file_name+'\n')
        else:
            print('skip:'+file_name)
    except Exception as e:
        print(e)
        print('code:'+code)


if __name__ == "__main__":
    all_stock_list = get_all_stocks_index()
    now = datetime.datetime.now()
    datetime.datetime.strftime(now, "%Y%m%d")
    data_save_path = "data/"+datetime.datetime.strftime(now, "%Y%m%d")+"/"
    isExists=os.path.exists(data_save_path)
    if not isExists:
        os.makedirs(data_save_path)
    i = 0
    for stock in all_stock_list[:]:
        i += 1
        print(i,'downloading ' + stock)
        save_path = os.path.join(data_save_path, stock + '.csv')
        download_data(stock, save_path)
    os._exit(0)

