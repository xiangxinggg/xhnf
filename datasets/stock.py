from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import stock
import loader

# nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test
def get_mnist():
	"""Retrieve the MNIST dataset and process the datasets."""
	batch_size = 128
	nb_classes = 10
	
	# input image dimensions
	img_rows, img_cols = 28, 28
	
	# the datasets, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	
	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)
	
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, nb_classes)
	y_test = keras.utils.to_categorical(y_test, nb_classes)
	
	return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


# nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test
def get_stock():
	"""Retrieve the STOCK dataset and process the datasets."""
	batch_size = 512
	nb_classes = 2
	
	# input image dimensions
	img_rows, img_cols = 16, 7
	
	# the datasets, split between train and test sets
	(x_train, y_train), (x_test, y_test) = stock.load_data()
	
	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)
	
# 	x_train = x_train.astype('float32')
# 	x_test = x_test.astype('float32')
# 	x_train /= 255
# 	x_test /= 255
# 	print("*********** y_test:", y_test)
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, nb_classes)
	y_test = keras.utils.to_categorical(y_test, nb_classes)
# 	print("######## y_test:", y_test)
	
	return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_stock1():
	"""Retrieve the STOCK dataset and process the datasets."""
	batch_size = 1024
	nb_classes = 2
	
	# input image dimensions
	img_rows, img_cols = 128, 5
	
	# the datasets, split between train and test sets
	(x_train, y_train), (x_test, y_test) = loader.load_data()
	
	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)
	
# 	x_train = x_train.astype('float32')
# 	x_test = x_test.astype('float32')
# 	x_train /= 255
# 	x_test /= 255
# 	print("*********** y_test:", y_test)
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, nb_classes)
	y_test = keras.utils.to_categorical(y_test, nb_classes)
# 	print("######## y_test:", y_test)
	
	return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def main():
# 	dataset = 'mnist'
	dataset = 'stock'
	
	if dataset == 'cifar10':
		pass
# 		nb_classes, batch_size, input_shape, x_train, \
# 			x_test, y_train, y_test = get_cifar10()
	elif dataset == 'mnist':
		nb_classes, batch_size, input_shape, x_train, \
			x_test, y_train, y_test = get_mnist()
	elif dataset == 'stock':
		nb_classes, batch_size, input_shape, x_train, \
			x_test, y_train, y_test = get_stock()
	
	print("========================>>>")
	print('dataset:',dataset)
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	print("========================<<<")

if __name__ == '__main__':
    main()
