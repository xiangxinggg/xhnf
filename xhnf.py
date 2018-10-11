# --*-- coding:utf-8 --*--
import keras
from configs.configs import Configs
from datasets.datasets import get_data, get_predict_data, get_test_data
from models.model import get_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
# seed = 7
# np.random.seed(seed)

class XHNF (object):
    def __init__(self):
        self.config = None
        self.data = None
        self.model = None
        self.predict_data = None

    def init_config(self):
#         self.config = Configs(model='default', dataset='default', epochs=10, batch_size=128)
#         self.config = Configs(model='resnet', dataset='cifar10', epochs=10, batch_size=128)
#         self.config = Configs(model='resnet', dataset='mnist', epochs=10, batch_size=128)
        self.config = Configs(model='resnet', dataset='stock', epochs=10000, batch_size=128)
    
    def init_data(self):
        self.data = get_data(self.config.dataset)

    def init_predict_data(self):
        self.predict_data = get_predict_data(self.config.dataset)
        self.data = self.predict_data

    def init_test_data(self):
        self.predict_data = get_test_data(self.config.dataset)
        self.data = self.predict_data

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
        earlyStopping = EarlyStopping( monitor='val_loss', patience=20, verbose=0, mode='auto')
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            print("checkpoint_loaded")
        self.model.fit(self.data.x_train, self.data.y_train,
                  batch_size=self.config.batch_size,
                  epochs=self.config.epochs,
                  verbose=1,
                  validation_data=(self.data.x_test, self.data.y_test),
                  callbacks=[checkpoint, earlyStopping])
        score = self.model.evaluate(self.data.x_test, self.data.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def test(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
        filepath = self.getModelFileName()

        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            print("checkpoint_loaded")

        y_predict = self.model.predict(self.data.x_train)
        y_res = y_predict[:,1]-y_predict[:,0]
        y_res = y_res.reshape(y_res.shape[0],1)

        yt = np.delete(self.data.y_train, (0), axis=1)

        y = np.concatenate((self.data.y_test, y_res), axis=1)
        #y = np.concatenate((y, y_predict), axis=1)
        y = np.concatenate((y, yt), axis=1)
        #y = np.concatenate((y, self.predict_data.y_train), axis=1)
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
        y_predict = self.model.predict(self.predict_data.x_train)
        y_res = y_predict[:,1]-y_predict[:,0]
        y_res = y_res.reshape(y_res.shape[0],1)

        yt = np.delete(self.predict_data.y_train, (0), axis=1)

        y = np.concatenate((self.predict_data.y_test, y_res), axis=1)
        #y = np.concatenate((y, y_predict), axis=1)
        y = np.concatenate((y, yt), axis=1)
        #y = np.concatenate((y, self.predict_data.y_train), axis=1)
        print(y)

    def do_train(self):
        self.init()
        self.train_network()

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
