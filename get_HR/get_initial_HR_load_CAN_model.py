'''Trains CNN rPPG detector
'''
#%%
from __future__ import print_function
import json
import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import keras
#np.random.seed(1337)  # for reproducibility
import h5py
#from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, multiply
from keras.layers import Conv2D, AveragePooling2D, Lambda
#from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import os
from os.path import join, basename, dirname, exists

#%%

data_dir = '../Data/'
save_dir = 'HR_Result_loaded_CAN/'
model_path = 'model_HR_CAN.h5'
nb_filters1 = 32
nb_filters2 = 64
dropout_rate1 = 0.25
dropout_rate2 = 0.5
nb_dense = 128
cv_split = 0
nb_epoch = 48
batch_size = 128 #32 128
# input image dimensions
img_rows, img_cols = 36, 36
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

if os.path.exists(save_dir) == 0:
    os.mkdir(save_dir)
#%% load data
print('Loading data...')
signal_length = 861
Xtest = np.zeros((signal_length+1, img_rows, img_cols, 6),dtype=np.float32)

for i in range(1):
    f1 = h5py.File(data_dir + 'sample_CAN_input_data.mat')
    dXsub = np.transpose(np.array(f1["dXsub"]))
    Xtest[i*signal_length:(i+1)*signal_length,:,:,:] = dXsub
    f1.close()

input_shape = (img_rows, img_cols, 3)
print(Xtest.shape[0], 'test samples')

#%%
def masknorm(x):
    xsum = K.sum(K.sum(x, axis=1, keepdims=True), axis=2, keepdims=True)
    xshape = K.int_shape(x)
    return x/xsum*xshape[1]*xshape[2]*0.5
    
def masknorm_shape(input_shape):
    return input_shape

#%%
print('Build model...')
diff_input = Input(shape=input_shape)
rawf_input = Input(shape=input_shape)

d1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
d2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(d1)

r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
g1 = Lambda(masknorm, output_shape=masknorm_shape)(g1)
gated1 = multiply([d2, g1])

d3 = AveragePooling2D(pool_size)(gated1)
d4 = Dropout(dropout_rate1)(d3)

r3 = AveragePooling2D(pool_size)(r2)
r4 = Dropout(dropout_rate1)(r3)

d5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
d6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(d5)

r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
g2 = Lambda(masknorm, output_shape=masknorm_shape)(g2)
gated2 = multiply([d6, g2])

d7 = AveragePooling2D(pool_size)(gated2)
d8 = Dropout(dropout_rate1)(d7)

d9 = Flatten()(d8)
d10 = Dense(nb_dense, activation='tanh')(d9)
d11 = Dropout(dropout_rate2)(d10)
out = Dense(1)(d11)

model = Model(inputs=[diff_input, rawf_input], outputs=out)

model.compile(loss='mean_squared_error', optimizer='adadelta')
model.load_weights(model_path)

#%% save predicted physiological signal
yptest = model.predict([Xtest[:,:,:,:3], Xtest[:,:,:,-3:]], batch_size=batch_size, verbose=0)
scipy.io.savemat(save_dir + '/yptest' + str(1) + '.mat', mdict={'yptest': yptest})

#%% save attention masks
layer_name = 'conv2d_5'
intermediate_layer_model1 = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
mask1 = intermediate_layer_model1.predict([Xtest[:,:,:,:3], Xtest[:,:,:,-3:]], batch_size=batch_size, verbose=0)
scipy.io.savemat(save_dir + '/mask1.mat', mdict={'mask1': mask1})

#%%
layer_name = 'conv2d_10'
intermediate_layer_model2 = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
mask2 = intermediate_layer_model2.predict([Xtest[:,:,:,:3], Xtest[:,:,:,-3:]], batch_size=batch_size, verbose=0)
scipy.io.savemat(save_dir + '/mask2.mat', mdict={'mask2': mask2})
