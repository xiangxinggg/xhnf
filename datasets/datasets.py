# --*-- coding:utf-8 --*--
'''
Created on 2018.10.3

@author: xiangxing
'''
from datasets.mnist import get_mnist
from datasets.cifar10 import get_cifar10
from datasets.stock import get_stock, get_predict_stock, get_test_stock
from entitys.entitys import Entitys

def get_data( name, start_date, end_date):
    entitys = None
    if name == 'mnist' or name == 'default':
        nb_classes, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist()
        entitys = Entitys(input_shape, nb_classes, x_train, y_train, x_test, y_test)
    elif name == 'cifar10':
        nb_classes, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
        entitys = Entitys(input_shape, nb_classes, x_train, y_train, x_test, y_test)
    elif name == 'stock':
        nb_classes, input_shape, x_train, \
            x_test, y_train, y_test = get_stock(start_date, end_date)
        entitys = Entitys(input_shape, nb_classes, x_train, y_train, x_test, y_test)
    
    return entitys


def get_predict_data( name, start_date, end_date):
    entitys = None
    if name == 'mnist' or name == 'default':
        pass
    elif name == 'cifar10':
        pass
    elif name == 'stock':
        nb_classes, input_shape, x_train, \
            x_test, y_train, y_test = get_predict_stock(start_date, end_date)
        entitys = Entitys(input_shape, nb_classes, x_train, y_train, x_test, y_test)
    
    return entitys


def get_test_data( name, start_date, end_date):
    entitys = None
    if name == 'mnist' or name == 'default':
        pass
    elif name == 'cifar10':
        pass
    elif name == 'stock':
        nb_classes, input_shape, x_train, \
            x_test, y_train, y_test = get_test_stock(start_date, end_date)
        entitys = Entitys(input_shape, nb_classes, x_train, y_train, x_test, y_test)
    
    return entitys