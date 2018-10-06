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

def load_csv(last_train_date, total_ahead_dates, fname, col_start=2, row_start=1, delimiter=",", dtype=dtypes.float32):
  data = np.genfromtxt(fname, delimiter=delimiter,skip_header=row_start, dtype=str)
  print('row shape:', data.shape)
#   print('data[0]',data[0])
#   for _ in range(row_start):
#     data = np.delete(data, (0), axis=0)

  myDate = datetime.datetime.strptime(last_train_date,'%Y%m%d')

  pliteIdx = -1
  for idx in range(data.shape[0]):
    dateStr = str(data[idx][0])
#     print('str:', dateStr)
    if dateStr.find('-') > 0 :
        date = datetime.datetime.strptime(dateStr,'%Y-%m-%d')
    else:
        date = datetime.datetime.strptime(dateStr,'%Y%m%d')

    if myDate < date and pliteIdx == -1:
        pliteIdx = idx
    #print("date:", date)
  if pliteIdx != -1:
      ignore_start = pliteIdx - total_ahead_dates

  if pliteIdx > 0:
    for _ in range(ignore_start):
      data = np.delete(data, (0), axis=0)
  l = data.shape[0]
  if l > total_ahead_dates:
    for _ in range(l, total_ahead_dates, -1):
      data = np.delete(data, (total_ahead_dates), axis=0)
  #print("now len:",l)

#     print('**********reserve date*********')
#     for idx in range(data.shape[0]):
#         print(data[idx][0])
#     print('===================')
#     print('last_train_date:',last_train_date)
#     print('total_ahead_dates:',total_ahead_dates)
#     print('shape:', data.shape)

  for _ in range(col_start):
    data = np.delete(data, (0), axis=1)
  # print(np.transpose(datasets))
  data = data.astype(np.float32)
#   print('data[0]',data[0])
  return data

# stock datasets loading
def load_stock(last_train_date, total_ahead_dates=360, pre_dates=3, path="data"+os.path.sep+"daily" \
               , moving_window=128, train_test_ratio=4.0):
  # process a single file's datasets into usable arrays
  def process_data(data, pre_dates):
    print('data.shape',data.shape)
    columns = data.shape[1]
    stock_set = np.zeros([0,moving_window,columns])
    label_set = np.zeros([0,1])
    for idx in range(data.shape[0] - (moving_window + pre_dates)):
      ss = np.expand_dims(data[range(idx,idx+(moving_window)),:], axis=0)
      stock_set = np.concatenate((stock_set, ss), axis=0)

      if data[idx+(moving_window+pre_dates),3] > data[idx+(moving_window),3]:
        lbl = [[1.0]]
#         print(data[idx+(moving_window+pre_dates),3],data[idx+(moving_window),3],'true')
      else:
        lbl = [[0.0]]
#         print(data[idx+(moving_window+pre_dates),3],data[idx+(moving_window),3],'false')
      label_set = np.concatenate((label_set, lbl), axis=0)
    return stock_set, label_set

  # read a directory of datasets
  stocks_set = None
  labels_set = np.zeros([0,1])
  ii = 0
  for dir_item in os.listdir(path):
    dir_item_path = os.path.join(path, dir_item)
    if os.path.isfile(dir_item_path):
      ii += 1
      print("index:",ii,"\t",dir_item_path)
      ss, ls = process_data(load_csv(last_train_date, total_ahead_dates,dir_item_path), pre_dates=3)
      if stocks_set is None:
          print('ss.shape:', ss.shape)
          stocks_set = np.zeros([0,moving_window,ss.shape[2]])
      stocks_set = np.concatenate((stocks_set, ss), axis=0)
      labels_set = np.concatenate((labels_set, ls), axis=0)

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
  train_stocks = stocks_set[train_test_idx:,:,:]
  train_labels = labels_set[train_test_idx:]
  test_stocks = stocks_set[:train_test_idx,:,:]
  test_labels = labels_set[:train_test_idx]

  return (train_stocks, train_labels), (test_stocks, test_labels)
