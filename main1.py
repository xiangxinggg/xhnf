# --*-- coding:utf-8 --*--

import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.models import Model
from datasets1 import get_mnist, get_stock, get_stock1
from keras.layers.convolutional import Conv2D,MaxPooling2D
#from model import get_model
from model_resnet import get_model
import numpy as np

# seed = 7
# np.random.seed(seed)
# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def do_all(nb_classes, epochs, input_shape, batch_size, x_train, y_train,x_test, y_test):
# 	model = Sequential()
# 	model.add(Conv2D(32, kernel_size=(3, 3),
# 	                 activation='relu',
# 	                 input_shape=input_shape))
# 	model.add(Conv2D(64, (3, 3), activation='relu'))
# #	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	model.add(Dropout(0.25))
# 	model.add(Flatten())
# 	model.add(Dense(128, activation='relu'))
# 	model.add(Dropout(0.5))
# 	model.add(Dense(nb_classes, activation='softmax'))

	model = get_model(nb_classes, input_shape)
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])
	
	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


def main():
# 	dataset = 'cifar10'
#	dataset = 'mnist'
#	dataset = 'stock'
	dataset = 'stock1'
	
	epochs = 10000

	if dataset == 'cifar10':
		pass
#  		nb_classes, batch_size, input_shape, x_train, \
#  			x_test, y_train, y_test = get_cifar10()
	elif dataset == 'mnist':
		nb_classes, batch_size, input_shape, x_train, \
			x_test, y_train, y_test = get_mnist()
	elif dataset == 'stock':
		nb_classes, batch_size, input_shape, x_train, \
			x_test, y_train, y_test = get_stock()
	elif dataset == 'stock1':
		nb_classes, batch_size, input_shape, x_train, \
			x_test, y_train, y_test = get_stock1()
	
	do_all(nb_classes, epochs, input_shape, batch_size, x_train, y_train, x_test, y_test);


if __name__ == '__main__':
    main()
