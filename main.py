# --*-- coding:utf-8 --*--

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from datasets import get_cifar10, get_mnist
from keras.layers.convolutional import Conv2D,MaxPooling2D
import numpy as np
seed = 7
np.random.seed(seed)
# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def do_all(nb_classes, epochs, input_shape, batch_size, x_train, y_train,x_test, y_test):
	model = Sequential()
	model.add(Dense(64, input_shape=input_shape))
	model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(nb_classes, activation='softmax'))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	model.fit(x_train, y_train, \
  	batch_size=batch_size, \
  	epochs=epochs,  \
  	verbose=0, \
  	validation_data=(x_test, y_test), \
  	callbacks=[early_stopper])

	score = model.evaluate(x_test, y_test, verbose=0)
	print("loss:",score[0])
	print("accuracy:",score[1])

def main():
	dataset = 'cifar10'
	#dataset = 'mnist'
	#dataset = 'stock'
	
	epochs = 100000

	if dataset == 'cifar10':
		nb_classes, batch_size, input_shape, x_train, \
			x_test, y_train, y_test = get_cifar10()
	elif dataset == 'mnist':
		nb_classes, batch_size, input_shape, x_train, \
			x_test, y_train, y_test = get_mnist()
	elif dataset == 'stock':
		pass
	
	do_all(nb_classes, epochs, input_shape, batch_size, x_train, y_train, x_test, y_test);


if __name__ == '__main__':
    main()
