# --*-- coding:utf-8 --*--

'''
Created on 2018.10.3

@author: xiangxing
'''

class Configs (object):
    def __init__(self, model='default', dataset='default', epochs=10, batch_size=128):
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        print('model:',model,';dataset:',dataset,';epochs:',epochs,';batch_size:',batch_size)

    def get_dataset(self):
        return self.dataset
    def get_epochs(self):
        return self.epochs
    def get_batch_size(self):
        return self.batch_size
    def get_model(self):
        return self.model