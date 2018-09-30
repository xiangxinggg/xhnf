# --*-- coding:utf-8 --*--

import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from datasets1 import get_mnist, get_stock, get_stock1
from keras.layers.convolutional import Conv2D,MaxPooling2D
import numpy as np

def get_model(nb_classes, input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    
    return model