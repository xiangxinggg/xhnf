# --*-- coding:utf-8 --*--
import keras
from configs.configs import Configs
from datasets.datasets import get_data
from models.model import get_model

class XHNF (object):
    def __init__(self):
        self.config = None
        self.data = None
        self.model = None

    def init_config(self):
#         self.config = Configs(model='default', dataset='default', epochs=10, batch_size=128)
#         self.config = Configs(model='resnet', dataset='cifar10', epochs=10, batch_size=128)
        self.config = Configs(model='resnet', dataset='stock', epochs=10, batch_size=128)
    
    def init_data(self):
        self.data = get_data(self.config.dataset)

    def init_model(self):
        self.model = get_model(self.config.model, self.data.input_shape, self.data.nb_classes)

    def init(self):
        self.init_config()
        self.init_data()
        self.init_model()
    
    def train_network(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
        
        self.model.fit(self.data.x_train, self.data.y_train,
                  batch_size=self.config.batch_size,
                  epochs=self.config.epochs,
                  verbose=1,
                  validation_data=(self.data.x_test, self.data.y_test))
        score = self.model.evaluate(self.data.x_test, self.data.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
    def do_all(self):
        self.init()
        self.train_network()
    
def main():
    print("Start XHNF.")
    xhnf = XHNF()
    xhnf.do_all()
    print("All Done.")

if __name__ == '__main__':
    main()
