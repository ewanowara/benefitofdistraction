'''Trains CNN respiration detector
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
data_dir = 'AFRL/'
save_dir = 'BR_Result_loaded_CAN/'
model_path = 'model_BR_CAN.h5'
nb_filters1 = 32
nb_filters2 = 64
dropout_rate1 = 0.25
dropout_rate2 = 0.5
nb_dense = 32
cv_split = 0
nb_epoch = 48
nb_task = 2 # 1 to 6       
#%%
batch_size = 128 #32 128
#nb_classes = 10
nb_epoch = nb_epoch #48 16

# input image dimensions
img_rows, img_cols = 36, 36
# number of convolutional filters to use
nb_filters1 = nb_filters1
nb_filters2 = nb_filters2
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# dropout rates
dropout_rate1 = dropout_rate1
dropout_rate2 = dropout_rate2
# number of dense units
nb_dense = nb_dense
nb_task = nb_task
cv_split = cv_split

if os.path.exists(save_dir) == 0:
    os.mkdir(save_dir)
#%% load data
# code written for the AFRL dataset. Please download the AFRL dataset (citation below) or another large video dataset with ground truth pulse signals.
# @inproceedings{estepp2014recovering,
#   title={Recovering pulse rate during motion artifact with a multi-imager array for non-contact imaging photoplethysmography},
#   author={Estepp, Justin R and Blackford, Ethan B and Meier, Christopher M},
#   booktitle={2014 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
#   pages={1462--1469},
#   year={2014},
#   organization={IEEE}
# }
#%%
print('Loading data...')
Xtest = np.zeros((7200*25*2, img_rows, img_cols, 6),dtype=np.float32)
ytest = np.zeros((7200*25*2, 1),dtype=np.float32)
# loop through different test subjects with 5-fold validation
subTest = np.array([1,2,3,4,5])

for i in range(1):
    f1 = h5py.File(data_dir + '/P' + str(subTest[i]) + 'T' + str(nb_task) + 'VideoB2.mat')
    dXsub = np.transpose(np.array(f1["dXsub"]))
    Xtest[i*36000:(i+1)*36000,:,:,:] = dXsub
    f1.close()
    print('P' + str(subTest[i]) + 'T' + str(nb_task) + ' loaded')
for i in range(1):
    f1 = h5py.File(data_dir + '/P' + str(subTest[i]) + 'T' + str(nb_task+6) + 'VideoB2.mat')
    dXsub = np.transpose(np.array(f1["dXsub"]))
    Xtest[(i+5)*36000:(i+6)*36000,:,:,:] = dXsub
    f1.close()
    print('P' + str(subTest[i]) + 'T' + str(nb_task+6) + ' loaded')    


input_shape = (img_rows, img_cols, 3)

#Xtrain = Xtrain.astype('float32')
#Xtest = Xtest.astype('float32')
#Xtrain /= 255
#Xtest /= 255
# print('X_train shape:', Xtrain.shape)
# print(Xtrain.shape[0], 'train samples')
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
scipy.io.savemat(save_dir + '/yptest.mat', mdict={'yptest': yptest})

 
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
