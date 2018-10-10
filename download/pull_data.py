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

if __name__ == "__main__":
    all_stock_list = get_all_stocks_index()
    now = datetime.datetime.now()
    datetime.datetime.strftime(now, "%Y%m%d")
    data_save_path = "data/"+datetime.datetime.strftime(now, "%Y%m%d")+"/"
    isExists=os.path.exists(data_save_path)
    if not isExists:
        os.makedirs(data_save_path)

    for stock in all_stock_list[:]:
        print('reading data from ' + stock)
        df = get_data(code = stock)
        save_path = os.path.join(data_save_path, stock + '.csv')
        df.to_csv(save_path, encoding="utf_8_sig")
    os._exit(0)

