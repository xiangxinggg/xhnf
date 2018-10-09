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
np.set_printoptions(threshold=np.nan)


class Loader (object):
    def __init__(self):
        pass
    
    def load_csv(self, last_train_date, total_ahead_dates, fname \
                 , col_start=1, row_start=1, delimiter=",", dtype=dtypes.float32):
        data = np.genfromtxt(fname, delimiter=delimiter, skip_header=row_start, dtype=str)
        print('row shape:', data.shape)
        #   print('data[0]',data[0])
        #   for _ in range(row_start):
        #     data = np.delete(data, (0), axis=0)
        
        myDate = datetime.datetime.strptime(last_train_date, '%Y%m%d')
        
        pliteIdx = -1
        for idx in range(data.shape[0]):
          dateStr = str(data[idx][0])
        #     print('str:', dateStr)
          if dateStr.find('-') > 0 :
              date = datetime.datetime.strptime(dateStr, '%Y-%m-%d')
          else:
              date = datetime.datetime.strptime(dateStr, '%Y%m%d')
        
          if myDate > date and pliteIdx == -1:
              pliteIdx = idx
          # print("date:", date)
        if pliteIdx != -1:
            ignore_start = pliteIdx - total_ahead_dates
        
        if pliteIdx > 0:
          for _ in range(ignore_start):
            data = np.delete(data, (0), axis=0)
        l = data.shape[0]
        if l > total_ahead_dates:
          for _ in range(l, total_ahead_dates, -1):
            data = np.delete(data, (total_ahead_dates), axis=0)
        # print("now len:",l)
        
        #   print('**********reserve date*********')
        #   for idx in range(data.shape[0]):
        #     print(data[idx][0])
        #   print('===================')
        #   print('last_train_date:', last_train_date)
        #   print('total_ahead_dates:', total_ahead_dates)
        #   print('shape:', data.shape)
        date = np.split(data, [1,data.shape[1]], axis=1)[0]
        for _ in range(col_start):
          data = np.delete(data, (0), axis=1)
        # print(np.transpose(datasets))
        data = data.astype(np.float32)
        data = data[::-1]
        date = date[::-1]
        #   print('data[0]',data[0])
        return data, date

  # process a single file's datasets into usable arrays
    def process_data(self, data, date, code, pre_dates, moving_window, p_call):
      print('data.shape', data.shape)
      columns = data.shape[1]
      stock_set = np.zeros([0, moving_window, columns])
      label_set = np.zeros([0, 1])
      predict_set = np.zeros([0, 2])
      for idx in range(data.shape[0] - (moving_window + pre_dates)):
          ss = np.expand_dims(data[range(idx, idx + (moving_window)), :], axis=0)
          stock_set = np.concatenate((stock_set, ss), axis=0)
          label_set = p_call(data, idx, moving_window, pre_dates, label_set)
          dbl = [[date[idx+moving_window,0],code]]
          predict_set = np.concatenate((predict_set, dbl), axis=0)
      return stock_set, label_set, predict_set

    def read_raw_data(self, p_call, last_train_date, total_ahead_dates, pre_dates, path, moving_window):
      # read a directory of datasets
      stocks_set = None
      labels_set = np.zeros([0, 1])
      predict_set = np.zeros([0, 2])
      ii = 0
      for dir_item in os.listdir(path):
        dir_item_path = os.path.join(path, dir_item)
        if os.path.isfile(dir_item_path):
          ii += 1
          print("index:", ii, "\t", dir_item_path)
          code = dir_item[:6]
          data,date = self.load_csv(last_train_date, total_ahead_dates, dir_item_path)
          ss, ls, ps = self.process_data(data, date, code, pre_dates, moving_window, p_call)
          if stocks_set is None:
              print('ss.shape:', ss.shape)
              stocks_set = np.zeros([0, moving_window, ss.shape[2]])
          stocks_set = np.concatenate((stocks_set, ss), axis=0)
          labels_set = np.concatenate((labels_set, ls), axis=0)
          predict_set = np.concatenate((predict_set, ps), axis=0)
      return (stocks_set, labels_set, predict_set)

    def read_train_data(self, p_call, last_train_date, total_ahead_dates, pre_dates, path, moving_window, train_test_ratio):
      (stocks_set, labels_set, _) = self.read_raw_data(p_call, last_train_date, total_ahead_dates, pre_dates, path, moving_window)
      
      # shuffling the datasets
      perm = np.arange(labels_set.shape[0])
      np.random.shuffle(perm)
      stocks_set = stocks_set[perm]
      labels_set = labels_set[perm]
    
      # normalize the datasets
      stocks_set_ = np.zeros(stocks_set.shape)
      for i in range(len(stocks_set)):
        min = stocks_set[i].min(axis=0)
        max = stocks_set[i].max(axis=0)
        stocks_set_[i] = (stocks_set[i] - min) / (max - min)
      stocks_set = stocks_set_
      # labels_set = np.transpose(labels_set)
    
      # selecting 1/5 for testing, and 4/5 for training
      train_test_idx = int((1.0 / (train_test_ratio + 1.0)) * labels_set.shape[0])
      train_stocks = stocks_set[train_test_idx:, :, :]
      train_labels = labels_set[train_test_idx:]
      test_stocks = stocks_set[:train_test_idx, :, :]
      test_labels = labels_set[:train_test_idx]
      return (train_stocks, train_labels), (test_stocks, test_labels)


    def read_predict_data(self, p_call, last_train_date, total_ahead_dates, pre_dates, path, moving_window, train_test_ratio):
      (stocks_set, labels_set, predict_set) = self.read_raw_data(p_call, last_train_date, total_ahead_dates, pre_dates, path, moving_window)

      # normalize the datasets
      stocks_set_ = np.zeros(stocks_set.shape)
      for i in range(len(stocks_set)):
        min = stocks_set[i].min(axis=0)
        max = stocks_set[i].max(axis=0)
        stocks_set_[i] = (stocks_set[i] - min) / (max - min)
      stocks_set = stocks_set_
      # labels_set = np.transpose(labels_set)

      return (stocks_set, labels_set), (stocks_set, predict_set)
  
# stock datasets loading
def load_stock(p_call, last_train_date, total_ahead_dates=360, pre_dates=3, path="data" + os.path.sep + "daily" \
               , moving_window=128, train_test_ratio=4.0):
    loader = Loader()
    return loader.read_train_data(p_call, last_train_date, total_ahead_dates, pre_dates, path, moving_window, train_test_ratio)


def load_predict_stock(p_call, last_train_date, total_ahead_dates=360, pre_dates=3, path="data" + os.path.sep + "daily" \
               , moving_window=128, train_test_ratio=4.0):
    loader = Loader()
    return loader.read_predict_data(p_call, last_train_date, total_ahead_dates, pre_dates, path, moving_window, train_test_ratio)