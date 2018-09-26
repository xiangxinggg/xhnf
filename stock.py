
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from ..utils.data_utils import get_file
import numpy as np
import os
from tensorflow.python.framework import dtypes

# np.seterr(divide='ignore', invalid='ignore')

def load_csv(fname, col_start=2, row_start=1, delimiter=",", dtype=dtypes.float32):
    data = np.genfromtxt(fname, delimiter=delimiter)
    for _ in range(col_start):
        data = np.delete(data, (0), axis=1)
    for _ in range(row_start):
        data = np.delete(data, (0), axis=0)
    # print(np.transpose(data))
    return data

#1datetime,            2code,    3open,    4close,    5high,    6low,    7vol,       8amount,       9p_change
#2018-09-07 15:00:00,  600165,   4.48,     4.48,      4.48,     4.48,    9349.0,     4188352.0,     0.0
def process_data(data, moving_window, columns):
    stock_set = np.zeros([0,moving_window,columns])
    label_set = np.zeros([0,1])
    pred_date = 5
#     print(data)
    num_groups = data.shape[0] - (moving_window + pred_date)
    print('num_groups:', num_groups)
    num_groups = 5
    for idx in range(num_groups):
        stock_set = np.concatenate((stock_set, np.expand_dims(data[range(idx+pred_date,idx+pred_date+(moving_window)),:], axis=0)), axis=0)
#         print(stock_set)
#         print("data1:",data[idx,1],"data2:",data[idx+pred_date,1]);
    
        if data[idx,0] > data[idx+pred_date,0]:
            lbl = [[1.0]]
        else:
            lbl = [[0.0]]
        label_set = np.concatenate((label_set, lbl), axis=0)
        # label_set = np.concatenate((label_set, np.array([data[idx+(moving_window+5),3] - data[idx+(moving_window),3]])))
        # print(stock_set.shape, label_set.shape)
#     print(stock_set)
#     print("===========")
#     print(label_set)
    return stock_set, label_set

def load_stock_data( moving_window=16, columns=7):
    path = "data"
    stocks_set = np.zeros([0,moving_window,columns])
    labels_set = np.zeros([0,1])
    for dir_item in os.listdir(path):
        dir_item_path = os.path.join(path, dir_item)
        if os.path.isfile(dir_item_path):
            print(dir_item_path)
            ss, ls = process_data(load_csv(dir_item_path), moving_window, columns)
            stocks_set = np.concatenate((stocks_set, ss), axis=0)
            labels_set = np.concatenate((labels_set, ls), axis=0)
#     print(stocks_set)
#     print(labels_set)
    return (stocks_set, labels_set)

def load_data(train_test_ratio=4.0):
    """Loads the stock dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
#     path = get_file(path,
#                     origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
#                     file_hash='8a61469f7ea1b51cbae51d4f78837e45')
#     f = np.load(path)
#     x_train, y_train = f['x_train'], f['y_train']
#     x_test, y_test = f['x_test'], f['y_test']
#     f.close()

    stocks_set, labels_set = load_stock_data()
      # shuffling the data
    perm = np.arange(labels_set.shape[0])
    np.random.shuffle(perm)
    stocks_set = stocks_set[perm]
    labels_set = labels_set[perm]
    
#     # normalize the data
#     stocks_set_ = np.zeros(stocks_set.shape)
#     for i in range(len(stocks_set)):
#         min = stocks_set[i].min(axis=0)
#         max = stocks_set[i].max(axis=0)
#         print("i:",i,";min:",min,";max:",max)
#         stocks_set_[i] = (stocks_set[i] - min) / (max - min)
# #         print("done.")
#     stocks_set = stocks_set_
#     # labels_set = np.transpose(labels_set)
    
    # selecting 1/5 for testing, and 4/5 for training
    train_test_idx = int((1.0 / (train_test_ratio + 1.0)) * labels_set.shape[0])
    x_train = stocks_set[train_test_idx:,:,:]
    y_train = labels_set[train_test_idx:]
    x_test = stocks_set[:train_test_idx,:,:]
    y_test = labels_set[:train_test_idx]

    return (x_train, y_train), (x_test, y_test)


def main():
    print("stock")
#     load_stock_data()
    (x_train, y_train), (x_test, y_test) = load_data()
    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)

if __name__ == '__main__':
    main()
