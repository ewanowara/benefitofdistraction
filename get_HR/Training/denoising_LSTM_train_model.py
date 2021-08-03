'''Trains denoising LSTM 
'''
from random import seed
from random import uniform
from random import random
from numpy import array
from math import ceil
from math import log10
from math import sqrt
from numpy import argmax
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import Bidirectional
from functools import reduce  # omit on Python 2
import operator
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.io
import os
from os.path import join, basename, dirname, exists
import h5py
from scipy import signal
from scipy.io import loadmat
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
from keras.layers import Dropout
from keras.models import load_model
from scipy import fftpack
import keras.backend as K

#%%# The loading of the data is intended for the MMSE-HR dataset. Please download the dataset and use the following citation:
# @inproceedings{zhang2016multimodal,
#   title={Multimodal spontaneous emotion corpus for human behavior analysis},
#   author={Zhang, Zheng and Girard, Jeff M and Wu, Yue and Zhang, Xing and Liu, Peng and Ciftci, Umur and Canavan, Shaun and Reale, Michael and Horowitz, Andy and Yang, Huiyuan and others},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={3438--3446},
#   year={2016}
# }

# initialize 
# path to red, green, blue camera noise channels. 
# Here the .mat files in the red noise channel folder also contain 
# the physiological signal obtained from video with CAN and 
# the ground truth signal (for MMSE-HR the ground truth signal is the blood pressure waveform) 

# the signals should be upsampled to 30 fps

# when using this code, make sure you move files meant for training to a separate directory than the test set files, 
# otherwise change the code to train with leave-one-subject-out validation

# path to training set data
data_path_red_tr = 'Noise_masks_red_tr/'
data_path_green_tr = 'Noise_masks_green_tr/'
data_path_blue_tr = 'Noise_masks_blue_tr/'

# path to test set data
data_path_red_ts = 'Noise_masks_red_ts/'
data_path_green_ts = 'Noise_masks_green_ts/'
data_path_blue_ts = 'Noise_masks_blue_ts/'

# path to save the output signals
save_path = 'HR_Result_Train_LSTM/'
if os.path.exists(save_path) == 0:
	os.mkdir(save_path)

n_numbers = 4 # number of features - 1 signal + 3 noise estimates will be concatenated together
n_chars = 1 
n_batch = 32 # batch size
n_epoch = 11 # number of epochs
n_examples = 60 # length of the signals input to the LSTM

#%%# create the LSTM model
# if training LSTM from scratch:	
hidden_dim = 128
model = Sequential()
model.add(LSTM(hidden_dim, input_shape=(n_examples,n_numbers)))
model.add(RepeatVector(n_examples))
model.add(LSTM(hidden_dim, return_sequences=True))

model.add(TimeDistributed(Dense(n_chars, activation='tanh')))
model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

print(model.summary())

signal_length = 1000# len(noise_blue) # duration of the videos in the dataset in frames per second
n_samples = len(range(0,signal_length-(n_examples-1), int(n_examples/2))) # number of observations 

Xtrain1 = np.zeros((n_samples,n_examples),dtype=np.float32)
Xtrain2 = np.zeros((n_samples,n_examples),dtype=np.float32)
Xtrain3 = np.zeros((n_samples,n_examples),dtype=np.float32)
Xtrain4 = np.zeros((n_samples,n_examples),dtype=np.float32)
Xtrain = np.zeros((n_samples,n_examples,4),dtype=np.float32)
Ytrain = np.zeros((n_samples,n_examples),dtype=np.float32)

Xtest1 = np.zeros((n_samples,n_examples),dtype=np.float32)
Xtest2 = np.zeros((n_samples,n_examples),dtype=np.float32)
Xtest3 = np.zeros((n_samples,n_examples),dtype=np.float32)
Xtest4 = np.zeros((n_samples,n_examples),dtype=np.float32)
Xtest = np.zeros((n_samples,n_examples,4),dtype=np.float32)	
Ytest = np.zeros((n_samples,n_examples),dtype=np.float32)
#%%# load training data
loss_tr_save = []
epoch_save = []

i = -1
print('reading from path: ' + data_path_blue_tr)
for filename in os.listdir(data_path_blue_tr): 
	if filename.endswith(".mat"):
		i = i + 1	
		print('loading: ' + filename)
		data = h5py.File(os.path.join(data_path_blue_tr, filename))
		noise_blue = np.array(data["mask_noise_blue_mean2"]) # noise from the blue channel

		data = h5py.File(os.path.join(data_path_green_tr, filename))
		noise_green = np.array(data["mask_noise_green_mean2"]) # noise from the green channel

		data = h5py.File(os.path.join(data_path_red_tr, filename))
		noise_red = np.array(data["mask_noise_red_mean2"]) # noise from the red channel

		noisy_signal = np.array(data["yptest_sub2"]) # physiological signal from video output by CAN 
		GT_HR = np.array(data["bp_gt_resamp"]) # ground truth physiological signal (blood pressure wave for MMSE-HR)

		noise_red = np.transpose(noise_red)
		noise_green = np.transpose(noise_green)
		noise_blue = np.transpose(noise_blue)
		noisy_signal = np.transpose(noisy_signal)

		# normalize to -1 to 1 
		noise_red = noise_red / max(abs(noise_red))
		noise_green = noise_green / max(abs(noise_green))
		noise_blue = noise_blue / max(abs(noise_blue))
		noisy_signal = noisy_signal / max(abs(noisy_signal))
		GT_HR = GT_HR / max(abs(GT_HR))

		count = 0
		# reshape the 1D time signals, to #observations x 60 samples with 50% overlap
		for j in range(0,len(noisy_signal) - (n_examples-1), int(n_examples/2)):
			Xtrain1[count:(count+1), 0:n_examples] = np.transpose(noise_red[j:j+n_examples,:])
			Xtrain2[count:(count+1), 0:n_examples] = np.transpose(noise_green[j:j+n_examples,:])
			Xtrain3[count:(count+1), 0:n_examples] = np.transpose(noise_blue[j:j+n_examples,:])
			Xtrain4[count:(count+1), 0:n_examples] = np.transpose(noisy_signal[j:j+n_examples,:])
			count = count+1
				
		Xtrain[:,:,0] = Xtrain1
		Xtrain[:,:,1] = Xtrain2
		Xtrain[:,:,2] = Xtrain3
		Xtrain[:,:,3] = Xtrain4

	Xtrain = np.array(Xtrain)
	Ytrain = np.array(Ytrain)
	Ytrain = Ytrain.reshape(Ytrain.shape[0], Ytrain.shape[1], 1)

#%%# load test data
loss_ts_save = []
i = -1
for filename in os.listdir(data_path_blue_ts): 
	if filename.endswith(".mat"):
		i = i + 1	
		print('loading: ' + filename)
		data = h5py.File(os.path.join(data_path_blue_ts, filename))
		noise_blue = np.array(data["mask_noise_blue_mean2"]) # noise from the blue channel

		data = h5py.File(os.path.join(data_path_green_ts, filename))
		noise_green = np.array(data["mask_noise_green_mean2"]) # noise from the green channel

		data = h5py.File(os.path.join(data_path_red_ts, filename))
		noise_red = np.array(data["mask_noise_red_mean2"]) # noise from the red channel
 
		noisy_signal = np.array(data["yptest_sub2"]) # physiological signal from video output by CAN 
		GT_HR = np.array(data["bp_gt_resamp"]) # ground truth physiological signal (blood pressure wave for MMSE-HR)

		# normalize to -1 to 1 
		noise_red = noise_red / max(abs(noise_red))
		noise_green = noise_green / max(abs(noise_green))
		noise_blue = noise_blue / max(abs(noise_blue))
		noisy_signal = noisy_signal / max(abs(noisy_signal))
		GT_HR = GT_HR / max(abs(GT_HR))
		
		Xtest1 = np.zeros((n_samples,n_examples),dtype=np.float32)
		Xtest2 = np.zeros((n_samples,n_examples),dtype=np.float32)
		Xtest3 = np.zeros((n_samples,n_examples),dtype=np.float32)
		Xtest4 = np.zeros((n_samples,n_examples),dtype=np.float32)
		Xtest = np.zeros((n_samples,n_examples,4),dtype=np.float32)
		count = 0
		# reshape the 1D time signals, to #observations x 60 samples with 50% overlap
		for j in range(0,len(noisy_signal) - (n_examples-1), int(n_examples/2)):
			Xtest1[count:(count+1), 0:n_examples] = np.transpose(noise_red[j:j+n_examples,:])
			Xtest2[count:(count+1), 0:n_examples] = np.transpose(noise_green[j:j+n_examples,:])
			Xtest3[count:(count+1), 0:n_examples] = np.transpose(noise_blue[j:j+n_examples,:])
			Xtest4[count:(count+1), 0:n_examples] = np.transpose(noisy_signal[j:j+n_examples,:])
			count = count+1
				
		Xtest[:,:,0] = Xtest1
		Xtest[:,:,1] = Xtest2
		Xtest[:,:,2] = Xtest3
		Xtest[:,:,3] = Xtest4

	Xtest = np.array(Xtest)
	Ytest = np.array(Ytest)
	Ytest = Ytest.reshape(Ytest.shape[0], Ytest.shape[1], 1)

# train LSTM from scratch
for epoch in range(n_epoch):
	print('epoch #: ' + str(epoch))
	model.fit(Xtrain, Ytrain, epochs=1, batch_size=n_batch, verbose=0)
	predicted_tr = model.predict(Xtrain, verbose=0)
	expected_tr = Ytrain

	loss_tr = mean_squared_error(expected_tr[:,:,0], predicted_tr[:,:,0])
	loss_tr_save.append(loss_tr)
	epoch_save.append(epoch)

	print('training loss')
	print(loss_tr)
	
	# evaluate model on test set
	
	result = model.predict(Xtest, verbose=0)
	expected = Ytest
	predicted = result # this is the final output denoised physiological signal

	loss_ts = mean_squared_error(expected[:,:,0], predicted[:,:,0])
	loss_ts_save.append(loss_ts)
	print('test loss')
	print(loss_ts)	

# save predicted values for the last epoch
scipy.io.savemat(save_path + '/Xtrain.mat', mdict={'Xtrain': Xtest})
scipy.io.savemat(save_path + '/Ytrain.mat', mdict={'Ytrain': Ytest})
scipy.io.savemat(save_path + '/Xtest.mat', mdict={'Xtest': Xtest})
scipy.io.savemat(save_path + '/Ytest.mat', mdict={'Xtest': Ytest})
scipy.io.savemat(save_path + '/predicted.mat', mdict={'predicted': predicted})
scipy.io.savemat(save_path + '/loss_ts.mat', mdict={'loss_ts': loss_ts})
scipy.io.savemat(save_path + '/expected.mat', mdict={'expected': expected})

# save the trained model
model.save(save_path + '/model.h5')			


