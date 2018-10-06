# --*-- coding:utf-8 --*--
'''
Created on 2018.10.3

@author: xiangxing
'''
from keras import backend as K

def reshape_with_channels(x):
    # input image dimensions
    img_rows = x.shape[1]
    img_cols = 1
    channels = 1
    if len(x.shape) > 2 :
        img_cols = x.shape[2]
    if len(x.shape) > 3 :
        channels = x.shape[3]

    if K.image_data_format() == 'channels_first':
        x = x.reshape(x.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x = x.reshape(x.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)
    return (x, input_shape)
    