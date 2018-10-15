# --*-- coding:utf-8 --*--
import keras
from configs.configs import Configs
from datasets.datasets import get_data, get_predict_data, get_test_data
from models.model import get_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import datetime
# seed = 7
# np.random.seed(seed)

class XHNF (object):
    def __init__(self):
        self.config = None
        self.data = None
        self.model = None
        self.start_date = '20140101'
        self.end_date = '20170801'

    def init_config(self):
#         self.config = Configs(model='default', dataset='default', epochs=10, batch_size=128)
#         self.config = Configs(model='resnet', dataset='cifar10', epochs=10, batch_size=128)
#         self.config = Configs(model='resnet', dataset='mnist', epochs=10, batch_size=128)
        self.config = Configs(model='resnet152', dataset='stock', epochs=10000, batch_size=128)
    
    def init_data(self):
        self.data = get_data(self.config.dataset, self.start_date, self.end_date)

    def init_predict_data(self):
        now = datetime.datetime.now()
        end_date = datetime.datetime.strftime(now, "%Y%m%d")
        start = now+datetime.timedelta(days=-5)
        start_date = datetime.datetime.strftime(start, "%Y%m%d")
        self.data = get_predict_data(self.config.dataset, start_date, end_date)

    def init_test_data(self):
        now = datetime.datetime.now()
        end_date = datetime.datetime.strftime(now, "%Y%m%d")
        start = now+datetime.timedelta(days=-10)
        start_date = datetime.datetime.strftime(start, "%Y%m%d")
        self.data = get_test_data(self.config.dataset, start_date, end_date)

    def init_model(self):
        self.model = get_model(self.config.model, self.data.input_shape, self.data.nb_classes)

    def init(self):
        self.init_config()
        self.init_data()
        self.init_model()
    
    def init_predict(self):
        self.init_config()
        self.init_predict_data()
        self.init_model()

    def init_test(self):
        self.init_config()
        self.init_test_data()
        self.init_model()
        
    def train_network(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
        filepath = self.getModelFileName()
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss' \
                                     , save_weights_only=True,verbose=1,save_best_only=True, period=1)
        earlyStopping = EarlyStopping( monitor='val_loss', patience=100, verbose=0, mode='auto')
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            print("checkpoint_loaded")

        print('start training', self.end_date)
        self.model.fit(self.data.x_train, self.data.y_train,
              batch_size=self.config.batch_size,
              epochs=self.config.epochs,
              verbose=1,
              validation_data=(self.data.x_test, self.data.y_test),
              callbacks=[checkpoint, earlyStopping])
        score = self.model.evaluate(self.data.x_test, self.data.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def train(self):
        for i in range(1000):
            print('train run', i)
            if self.data == None :
                end = datetime.datetime.strptime(self.end_date, '%Y%m%d')
                now = datetime.datetime.now()
                if end < now :
                    end = end+datetime.timedelta(days=20)
                else:
                    print('all done, last train date ', self.end_date)
                    break
                end_date_str = datetime.datetime.strftime(end, "%Y%m%d")
                self.start_date = self.end_date
                self.end_date = end_date_str
                self.data = get_data(self.config.dataset, self.start_date, self.end_date)
                del self.model
                self.init_model()

            self.train_network()
            del self.data
            self.data = None

    def test(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
        filepath = self.getModelFileName()

        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            print("checkpoint_loaded")

        if self.data.x_train.shape[0] <= 0 :
            print("train shape:", self.data.x_train.shape)
            print("do not read any test date, so just exit.")
            return

        y_predict = self.model.predict(self.data.x_train)
        y_res = y_predict[:,1]-y_predict[:,0]
        y_res = y_res.reshape(y_res.shape[0],1)

        yt = np.delete(self.data.y_train, (0), axis=1)

        y = np.concatenate((self.data.y_test, y_res), axis=1)
        y = np.concatenate((y, yt), axis=1)
        print(y)
        
        score = self.model.evaluate(self.data.x_train, self.data.y_train, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def getModelFileName(self):
        records_dir = 'records' \
            + os.path.sep + self.config.get_model() \
            + os.path.sep + self.config.get_dataset()
        file_name = records_dir \
            + os.path.sep + 'model.h5'
        if os.path.exists(records_dir) == False:
            os.makedirs(records_dir)
        return file_name

    def do_predict(self):
        print('do predict.')
        self.init_predict()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
        filepath = self.getModelFileName()
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            print("checkpoint_loaded")

        if self.data.x_train.shape[0] <= 0 :
            print("train shape:", self.data.x_train.shape)
            print("do not read any predict date, so just exit.")
            return
        y_predict = self.model.predict(self.data.x_train)
        y_res = y_predict[:,1]-y_predict[:,0]
        y_res = y_res.reshape(y_res.shape[0],1)

        yt = np.delete(self.data.y_train, (0), axis=1)

        y = np.concatenate((self.data.y_test, y_res), axis=1)
        y = np.concatenate((y, yt), axis=1)
        print(y)

    def do_train(self):
        self.init()
        self.train()

    def do_test(self):
        self.init_test()
        self.test()

def main():
    print("Start XHNF.")
    xhnf = XHNF()
    xhnf.do_train()
    print("All Done.")

if __name__ == '__main__':
    main()
