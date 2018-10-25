# --*-- coding:utf-8 --*--

'''
Created on 2018��10��3��

@author: xiangxing
'''

class Entitys (object):
    nb_classes = 2
    epochs = 10
#     input_shape
    batch_size = 128
#     x_train
#     y_train
#     x_test
#     y_test
    def __init__(self, input_shape, nb_classes, x_train, y_train, x_test, y_test):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    def get_nb_classes(self):
        return self.nb_classes
    def get_epochs(self):
        return self.epochs
    def get_input_shape(self):
        return self.input_shape
    def get_batch_size(self):
        return self.batch_size
    def get_x_train(self):
        return None
    def get_y_train(self):
        return None
    def get_x_test(self):
        return None
    def get_y_test(self):
        return None