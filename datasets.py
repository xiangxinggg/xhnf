# --*-- coding:utf-8 --*--

import csv
import os
import numpy as np
from tensorflow.python.framework import dtypes
from keras.datasets import mnist, cifar10
from keras.utils.np_utils import to_categorical

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)



def load_csv(fname, col_start=1, row_start=1, delimiter=",", dtype=dtypes.float32):
  data = np.genfromtxt(fname, delimiter=delimiter)
  for _ in range(col_start):
    data = np.delete(data, (0), axis=1)
  for _ in range(row_start):
    data = np.delete(data, (0), axis=0)
  # print(np.transpose(data))
  return data

def process_data(data, moving_window=16, columns=8):
  stock_set = np.zeros([0,moving_window,columns])
  label_set = np.zeros([0,2])
  for idx in range(data.shape[0] - (moving_window + 5)):
    stock_set = np.concatenate((stock_set, np.expand_dims(data[range(idx+5,idx+5+(moving_window)),:], axis=0)), axis=0)
    print("data1:",data[idx,1],"data2:",data[idx+5,1]);

    if data[idx,1] > data[idx+5,1]:
      lbl = [[1.0, 0.0]]
    else:
      lbl = [[0.0, 1.0]]
    label_set = np.concatenate((label_set, lbl), axis=0)
    # label_set = np.concatenate((label_set, np.array([data[idx+(moving_window+5),3] - data[idx+(moving_window),3]])))
  # print(stock_set.shape, label_set.shape)
  return stock_set, label_set


def load_stock_data():
	path = "data"
	for dir_item in os.listdir(path):
		dir_item_path = os.path.join(path, dir_item)
		if os.path.isfile(dir_item_path):
			print(dir_item_path)
			ss, ls = process_data(load_csv(dir_item_path))
			#stocks_set = np.concatenate((stocks_set, ss), axis=0)
			#labels_set = np.concatenate((labels_set, ls), axis=0)


def get_stock():
    """Retrieve the STOCK dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


def main():
	load_stock_data()

if __name__ == '__main__':
    main()