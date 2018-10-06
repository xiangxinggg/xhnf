from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import os
from datasets.loader import load_stock

def get_stock():
	"""Retrieve the STOCK dataset and process the datasets."""
	nb_classes = 2
	last_train_date = '20130801'
	total_ahead_dates = 360
	pre_dates = 3
	path="data"+os.path.sep+"daily"

	# the datasets, split between train and test sets
	(x_train, y_train), (x_test, y_test) = load_stock(last_train_date, total_ahead_dates, pre_dates, path)

	# input image dimensions
	img_rows = x_train.shape[1]
	img_cols = x_train.shape[2]
	channels = 1

	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
	    input_shape = (channels, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
	    input_shape = (img_rows, img_cols, channels)
	
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
