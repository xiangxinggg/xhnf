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
from datasets.loader import load_stock

# open, Hight, close, low . ^ or V
def ohcl_callback(data, idx, moving_window, pre_dates, label_set):
# 	print('ohcl_callback')
	start = idx+moving_window
	end = start+pre_dates
	sum = 0
	for i in range(start, end):
		for n in range(4):
			sum += data[i+1,n]-data[i,n]
# 			print('i',i,'n',n,'+1',data[i+1,n], '0', data[i,n])
# 	print('sum', sum)
	if sum > 0 :
		lbl = [[1.0]]
	else:
		lbl = [[0.0]]
# 	if data[idx+(moving_window+pre_dates),0] > data[idx+(moving_window),0]:
# 	  lbl = [[1.0]]
# 	  print(data[idx+(moving_window+pre_dates),0],data[idx+(moving_window),0],'up')
# 	else:
# 	  lbl = [[0.0]]
# 	  print(data[idx+(moving_window+pre_dates),0],data[idx+(moving_window),0],'down')
	label_set = np.concatenate((label_set, lbl), axis=0)
	return label_set

def get_stock():
	"""Retrieve the STOCK dataset and process the datasets."""
	nb_classes = 2
	last_train_date = '20180711'
	total_ahead_dates = 200
	pre_dates = 3
	path="data"+os.path.sep+"daily"

	# the datasets, split between train and test sets
	(x_train, y_train), (x_test, y_test) = load_stock(ohcl_callback, last_train_date, total_ahead_dates, pre_dates, path)

	(x_train, input_shape) = reshape_with_channels(x_train)
	(x_test, _) = reshape_with_channels(x_test)
	
# 	x_train = x_train.astype('float32')
# 	x_test = x_test.astype('float32')
# 	x_train /= 255
# 	x_test /= 255
# 	print("*********** y_test:", y_test)
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, nb_classes)
	y_test = keras.utils.to_categorical(y_test, nb_classes)
# 	print("######## y_test:", y_test)
	
	return (nb_classes, input_shape, x_train, x_test, y_train, y_test)

def get_predict_stock():
	nb_classes = 2
	last_train_date = '20180711'
	total_ahead_dates = 200
	pre_dates = 3
	path="data"+os.path.sep+"daily"

	# the datasets, split between train and test sets
	(x_train, y_train), (x_test, y_test) = load_stock(ohcl_callback, last_train_date, total_ahead_dates, pre_dates, path)

	(x_train, input_shape) = reshape_with_channels(x_train)
	(x_test, _) = reshape_with_channels(x_test)
	
# 	x_train = x_train.astype('float32')
# 	x_test = x_test.astype('float32')
# 	x_train /= 255
# 	x_test /= 255
# 	print("*********** y_test:", y_test)
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, nb_classes)
	y_test = keras.utils.to_categorical(y_test, nb_classes)
# 	print("######## y_test:", y_test)
	
	return (nb_classes, input_shape, x_train, x_test, y_train, y_test)

def main():
	nb_classes, input_shape, x_train, \
		x_test, y_train, y_test = get_stock()
	
	print("========================>>>")
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	print("========================<<<")

if __name__ == '__main__':
    main()
