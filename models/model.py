# --*-- coding:utf-8 --*--
'''
Created on 2018.10.3

@author: xiangxing
'''
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.models import Model
from keras.layers.convolutional import Conv2D,MaxPooling2D
from models.resnet import get_resnet_model
import models.resnet2 as resnet2
import numpy as np

def get_default_model(input_shape, nb_classes):
    models = Sequential()
    models.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    models.add(Conv2D(64, (3, 3), activation='relu'))
#    models.add(MaxPooling2D(pool_size=(2, 2)))
    models.add(Dropout(0.25))
    models.add(Flatten())
    models.add(Dense(128, activation='relu'))
    models.add(Dropout(0.5))
    models.add(Dense(nb_classes, activation='softmax'))
    return models


def get_model( name, input_shape, nb_classes):
    if name == 'default':
        model = get_default_model(input_shape, nb_classes)
    elif name == 'resnet':
        model = get_resnet_model(input_shape, nb_classes)
    elif name == 'resnet18':
        model = resnet2.ResnetBuilder.build_resnet_18(input_shape, nb_classes)
    elif name == 'resnet152':
        model = resnet2.ResnetBuilder.build_resnet_152(input_shape, nb_classes)
    return model