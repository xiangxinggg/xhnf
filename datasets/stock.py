# --*-- coding:utf-8 --*--
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
from utils.utils import reshape_with_channels
import numpy as np
import os
import datetime
from datasets.loader import load_stock, load_predict_stock, load_test_stock

def get_nb_classes():
    return 4

def get_pre_dates():
    return 3

def get_moving_window():
    return 128

def get_date_str():
    now = datetime.datetime.now()
    date_str = datetime.datetime.strftime(now, "%Y%m%d")
    return date_str

# open, Hight, close, low . ^ or V
def ohcl_callback(data, idx, moving_window, pre_dates, label_set):
    start = idx+moving_window+1
    end = start+pre_dates-1
    total = 0
    for i in range(start, end):
        for n in range(4):
            total += data[i+1,n]-data[i,n]
            #print('i',i,'n',n,'+1',data[i+1,n], '0', data[i,n])

    per = (total/data[start, 0])*100/4
    #print("total", total, "per", per)

    nb_classes = get_nb_classes()
    if nb_classes == 2 :
        if per > 0 :
            lbl = [[1.0]]
        else:
            lbl = [[0.0]]
    elif nb_classes == 4 :
        if per >= 1 :
            lbl = [[1.0]]
        elif per >= 0 and per < 1:
            lbl = [[0.5]]
        elif per >= -1 and per < 0:
            lbl = [[-0.5]]
        else:
            lbl = [[-1.0]]
    else :
        if per > 0 :
            lbl = [[1.0]]
        else:
            lbl = [[0.0]]

    label_set = np.concatenate((label_set, lbl), axis=0)
    return label_set

def read_date_config():
    nb_classes = get_nb_classes()
    pre_dates = get_pre_dates()
    moving_window = get_moving_window()
    return nb_classes, pre_dates, moving_window, ohcl_callback

def get_stock(start_date = '20170101', end_date = '20180101'):
    """Retrieve the STOCK dataset and process the datasets."""
    nb_classes, pre_dates, moving_window, fun_callbak = read_date_config()
    path="data"+os.path.sep+"daily"

    # the datasets, split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_stock(fun_callbak, start_date, end_date, pre_dates, path, moving_window)

    (x_train, input_shape) = reshape_with_channels(x_train)
    (x_test, _) = reshape_with_channels(x_test)

    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    return (nb_classes, input_shape, x_train, x_test, y_train, y_test)

def get_predict_stock(start_date = '20170101', end_date = '20180101'):
    nb_classes, pre_dates, moving_window, fun_callbak = read_date_config()
    print('predict date:',end_date)
    path="data"+os.path.sep+"daily"

    # the datasets, split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_predict_stock(fun_callbak, start_date, end_date, pre_dates, path, moving_window)

    (x_train, input_shape) = reshape_with_channels(x_train)
    (x_test, _) = reshape_with_channels(x_test)

    y_train = keras.utils.to_categorical(y_train, nb_classes)

    return (nb_classes, input_shape, x_train, x_test, y_train, y_test)

def get_test_stock(start_date = '20170101', end_date = '20180101'):
    nb_classes, pre_dates, moving_window, fun_callbak = read_date_config()
    print('test date:',end_date)
    path="data"+os.path.sep+"daily"

    # the datasets, split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_test_stock(fun_callbak, start_date, end_date, pre_dates, path, moving_window)

    (x_train, input_shape) = reshape_with_channels(x_train)
    (x_test, _) = reshape_with_channels(x_test)

    y_train = keras.utils.to_categorical(y_train, nb_classes)

    return (nb_classes, input_shape, x_train, x_test, y_train, y_test)

def main():
    nb_classes, input_shape, x_train, x_test, y_train, y_test = get_stock()

    print("========================>>>")
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print("========================<<<")

if __name__ == '__main__':
    main()
