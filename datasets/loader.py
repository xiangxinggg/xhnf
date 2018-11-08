# --*-- coding:utf-8 --*--
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets import base
import math
import numpy as np
import pandas as pd
import functools as ft
import csv
import os
import datetime
import time
import sys
np.set_printoptions(threshold=np.nan)


class Loader (object):
    def __init__(self):
        pass
    
    def get_week_day(self, dateStr):
        if dateStr.find('-') > 0 :
            date = datetime.datetime.strptime(dateStr, '%Y-%m-%d')
        else:
            date = datetime.datetime.strptime(dateStr, '%Y%m%d')
        week_day = date.weekday()+1
        day = date.day
        return week_day, day

    def load_csv(self, fname, max_items, start_date='20170102', end_date='20180203', moving_window=128 \
                 , col_start=1, row_start=1, delimiter=",", dtype=dtypes.float32):
        data = np.genfromtxt(fname, delimiter=delimiter, skip_header=row_start, dtype=str)
        if data.shape[0] < moving_window:
            #print("can't read enough row data from cvs file",fname)
            #print('row shape:', data.shape)
            return None, None
        #print('row shape:', data.shape)
        #   print('data[0]',data[0])
        #   for _ in range(row_start):
        #     data = np.delete(data, (0), axis=0)
        
        start = datetime.datetime.strptime(start_date, '%Y%m%d')
        end = datetime.datetime.strptime(end_date, '%Y%m%d')
        for idx in range(data.shape[0]-1, -1, -1):
            dateStr = str(data[idx][0])
            #print('str:', dateStr)
            if dateStr.find('-') > 0 :
                date = datetime.datetime.strptime(dateStr, '%Y-%m-%d')
            else:
                date = datetime.datetime.strptime(dateStr, '%Y%m%d')
            if date > end:
                data = np.delete(data, (idx), axis=0)

        ignore_end = -1
        for idx in range(data.shape[0]):
            dateStr = str(data[idx][0])
            #print('str:', dateStr)
            if dateStr.find('-') > 0 :
                date = datetime.datetime.strptime(dateStr, '%Y-%m-%d')
            else:
                date = datetime.datetime.strptime(dateStr, '%Y%m%d')
            if date < start:
                ignore_end = idx + moving_window
                break
          
        if ignore_end != -1:
            l = data.shape[0]
            for _ in range(ignore_end, l):
                data = np.delete(data, (ignore_end), axis=0)

        if max_items != 0:
            l = data.shape[0]
            for _ in range(l, max_items, -1):
                data = np.delete(data, (max_items), axis=0)

        date = np.split(data, [1,data.shape[1]], axis=1)[0]
        for _ in range(col_start):
            data = np.delete(data, (0), axis=1)
        # print(np.transpose(datasets))
        data = data.astype(np.float32)
        data = data[::-1]
        date = date[::-1]
        #   print('data[0]',data[0])

        date_set = np.zeros([0, 2])
        for idx in range(date.shape[0]):
            #print(date[idx])
            week_day, day = self.get_week_day(date[idx][0])
            #print(week_day, day, date[idx][0])
            my_date = [[week_day, day]]
            date_set = np.concatenate((date_set, my_date), axis=0)

        data = np.concatenate((data, date_set), axis=1)
        #print("data.shape", data.shape, "date.shape", date.shape)
        return data, date

    # process a single file's datasets into usable arrays
    def process_data(self, data, date, code, pre_dates, moving_window, p_call, predict=False):
        #print('data.shape', data.shape)
        columns = data.shape[1]
        stock_set = np.zeros([0, moving_window, columns])
        label_set = np.zeros([0, 1])
        predict_set = np.zeros([0, 2])
        if predict == True:
            start = data.shape[0] - (moving_window + pre_dates)
            end = data.shape[0] - (moving_window)
            if end < 0:
                end = 0
            if start < 0:
                start = 0

            #print('start:',start,'end',end)
            for idx in range(start, end):
                ss = np.expand_dims(data[range(idx, idx + (moving_window)), :], axis=0)
                stock_set = np.concatenate((stock_set, ss), axis=0)
                lbl = [[1.0]]
                label_set = np.concatenate((label_set, lbl), axis=0)
                #label_set = p_call(data, idx, moving_window, pre_dates, label_set)
                dbl = [[date[idx+moving_window,0],"*"+code]]
                predict_set = np.concatenate((predict_set, dbl), axis=0)

        for idx in range(data.shape[0] - (moving_window + pre_dates)):
            ss = np.expand_dims(data[range(idx, idx + (moving_window)), :], axis=0)
            stock_set = np.concatenate((stock_set, ss), axis=0)
            label_set = p_call(data, idx, moving_window, pre_dates, label_set)
            dbl = [[date[idx+moving_window,0],"#"+code]]
            predict_set = np.concatenate((predict_set, dbl), axis=0)
        return stock_set, label_set, predict_set

    def read_raw_data(self, p_call, start_date, end_date, pre_dates, path, moving_window, predict=False, max_items=0):
        # read a directory of datasets
        stocks_set = None
        labels_set = np.zeros([0, 1])
        predict_set = np.zeros([0, 2])
        ii = 0
        files = os.listdir(path)
        total = len(files)
        #       max_items=moving_window+pre_dates+1
        for dir_item in files:
            dir_item_path = os.path.join(path, dir_item)
            if os.path.isfile(dir_item_path):
                ii += 1
                #print("index:", ii, "\t", dir_item_path)
                code = dir_item[:6]
                done = int(ii*50/total)
                sys.stdout.write("\r[%s%s] %d/%d,code:%s\r" % ('#' * done, ' ' * (50 - done),ii,total, code))
                sys.stdout.flush()
                data,date = self.load_csv(dir_item_path, max_items, start_date, end_date, moving_window)
                if data is None:
                    continue
                ss, ls, ps = self.process_data(data, date, code, pre_dates, moving_window, p_call, predict)
                if stocks_set is None:
                    #print('ss.shape:', ss.shape)
                    stocks_set = np.zeros([0, moving_window, ss.shape[2]])
                stocks_set = np.concatenate((stocks_set, ss), axis=0)
                labels_set = np.concatenate((labels_set, ls), axis=0)
                predict_set = np.concatenate((predict_set, ps), axis=0)
        return (stocks_set, labels_set, predict_set)

    def normalize_datasets(self, stocks_set):
        # normalize the datasets
        stocks_set_ = np.zeros(stocks_set.shape)
        for i in range(len(stocks_set)):
            vmin = stocks_set[i].min(axis=0)
            vmax = stocks_set[i].max(axis=0)
            stocks_set_[i] = (stocks_set[i] - vmin) / (vmax - vmin)
        stocks_set = stocks_set_
        # labels_set = np.transpose(labels_set)
        return stocks_set

    def read_train_data(self, p_call, start_date, end_date, pre_dates, path, moving_window, train_test_ratio):
        (stocks_set, labels_set, _) = self.read_raw_data(p_call, start_date, end_date, pre_dates, path, moving_window)

        # shuffling the datasets
        perm = np.arange(labels_set.shape[0])
        np.random.shuffle(perm)
        stocks_set = stocks_set[perm]
        labels_set = labels_set[perm]

        stocks_set = self.normalize_datasets(stocks_set)

        # selecting 1/5 for testing, and 4/5 for training
        train_test_idx = int((1.0 / (train_test_ratio + 1.0)) * labels_set.shape[0])
        train_stocks = stocks_set[train_test_idx:, :, :]
        train_labels = labels_set[train_test_idx:]
        test_stocks = stocks_set[:train_test_idx, :, :]
        test_labels = labels_set[:train_test_idx]
        return (train_stocks, train_labels), (test_stocks, test_labels)

    def read_predict_data(self, p_call, start_date, end_date, pre_dates, path, moving_window, train_test_ratio):
        (stocks_set, labels_set, predict_set) = self.read_raw_data(p_call, start_date, end_date, pre_dates, path, moving_window, predict=True)
        stocks_set = self.normalize_datasets(stocks_set)
        return (stocks_set, labels_set), (stocks_set, predict_set)

    def read_test_data(self, p_call, start_date, end_date, pre_dates, path, moving_window, train_test_ratio):
        (stocks_set, labels_set, predict_set) = self.read_raw_data(p_call, start_date, end_date, pre_dates, path, moving_window)
        stocks_set = self.normalize_datasets(stocks_set)
        return (stocks_set, labels_set), (stocks_set, predict_set)

# stock datasets loading
def load_stock(p_call, start_date, end_date, pre_dates=3, path="data" + os.path.sep + "daily" \
               , moving_window=128, train_test_ratio=4.0):
    loader = Loader()
    return loader.read_train_data(p_call, start_date, end_date, pre_dates, path, moving_window, train_test_ratio)

def load_predict_stock(p_call, start_date, end_date, pre_dates=3, path="data" + os.path.sep + "daily" \
               , moving_window=128, train_test_ratio=4.0):
    loader = Loader()
    return loader.read_predict_data(p_call, start_date, end_date, pre_dates, path, moving_window, train_test_ratio)

def load_test_stock(p_call, start_date, end_date, pre_dates=3, path="data" + os.path.sep + "daily" \
               , moving_window=128, train_test_ratio=4.0):
    loader = Loader()
    return loader.read_test_data(p_call, start_date, end_date, pre_dates, path, moving_window, train_test_ratio)
