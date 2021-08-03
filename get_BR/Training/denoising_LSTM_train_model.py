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
import h5py

from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
from keras.layers import Dropout

#%% code written for the AFRL dataset. Please download the AFRL dataset (citation below) or another large video dataset with ground truth breathing signals.
# @inproceedings{estepp2014recovering,
#   title={Recovering pulse rate during motion artifact with a multi-imager array for non-contact imaging photoplethysmography},
#   author={Estepp, Justin R and Blackford, Ethan B and Meier, Christopher M},
#   booktitle={2014 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
#   pages={1462--1469},
#   year={2014},
#   organization={IEEE}
# }

data_path_blue = 'Noise_masks_blue_BR/'
data_path_green = 'Noise_masks_green_BR/'
data_path_red = 'Noise_masks_red_BR/'

save_path = 'BR_Result_trained_LSTM/'

# train for all 6 motion tasks separately, and all motion tasks together
# define directories
for task in range(1,8): 
	print(task)
	if task == 1:
		# task 1
		tasks = np.array([1,7])
		n_subjects_tr = 40
		n_subjects_ts = 10
	elif task == 2:
		# task 2
		tasks = np.array([2,8])
		n_subjects_tr = 40
		n_subjects_ts = 10
	elif task == 3:
		# task 3
		tasks = np.array([3,9])
		n_subjects_tr = 40
		n_subjects_ts = 10
	elif task == 4:
		# task 4
		tasks = np.array([4,10])
		n_subjects_tr = 40
		n_subjects_ts = 10
	elif task == 5:
		# task 5
		n_subjects_tr = 40
		n_subjects_ts = 10
		tasks = np.array([5,11])
	elif task == 6:
		# task 6
		n_subjects_tr = 40
		n_subjects_ts = 10
		tasks = np.array([6,12])
	elif task == 7:
		# all
		tasks = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
		n_subjects_tr = 240
		n_subjects_ts = 60
	for split in range(0,5):
		# all tasks, split 0
		if split == 0:
			subNum_tr = np.array([6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,25,26,27])   
			subNum_ts = np.array([1,2,3,4,5])
		elif split == 1:
			# all tasks, split 1
			subNum_tr = np.array([1,2,3,4,5,11,12,13,14,15,16,17,18,20,21,22,23,25,26,27])   
			subNum_ts = np.array([6,7,8,9,10])
		elif split == 2:
			# all tasks, split 2
			subNum_tr = np.array([1,2,3,4,5,6,7,8,9,10,16,17,18,20,21,22,23,25,26,27])  
			subNum_ts = np.array([11,12,13,14,15])
		elif split == 3:
			# all tasks, split 3
			subNum_tr = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,25,26,27]) 
			subNum_ts = np.array([16,17,18,20,21])
		elif split == 4:
			# all tasks, split 4
			subNum_tr = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21])
			subNum_ts = np.array([22,23,25,26,27])
		
		if os.path.exists(save_path + '/task' + str(task) + '/split' + str(split) + '/model' + str(128) +  str(10) + '.h5') == 0:
			n_examples = 60 # each sequence length
			signal_length = 9000
			number_of_samples_per_subject = len(range(0,signal_length-(n_examples-1), int(n_examples/2)))
			n_samples_tr = n_subjects_tr*number_of_samples_per_subject # number of sequences in training set
			n_samples_ts = n_subjects_ts*number_of_samples_per_subject # number of test observations

			hidden_dim_list = np.array([128])

			n_numbers = 4 # number of features - 2 signals to be added together
			n_chars = 1
			n_batch = 32
			n_epoch = 11#300

			# load data
			Xtrain1 = np.zeros((n_samples_tr,n_examples),dtype=np.float32) 
			Xtrain2 = np.zeros((n_samples_tr,n_examples),dtype=np.float32)
			Xtrain3 = np.zeros((n_samples_tr,n_examples),dtype=np.float32)
			Xtrain4 = np.zeros((n_samples_tr,n_examples),dtype=np.float32)
			Xtrain = np.zeros((n_samples_tr,n_examples,4),dtype=np.float32)
			Xtest1 = np.zeros((n_samples_ts,n_examples),dtype=np.float32)
			Xtest2 = np.zeros((n_samples_ts,n_examples),dtype=np.float32)
			Xtest3 = np.zeros((n_samples_ts,n_examples),dtype=np.float32)
			Xtest4 = np.zeros((n_samples_ts,n_examples),dtype=np.float32)
			Xtest = np.zeros((n_samples_ts,n_examples,4),dtype=np.float32)
			Ytrain = np.zeros((n_samples_tr,n_examples),dtype=np.float32)
			Ytest = np.zeros((n_samples_ts,n_examples),dtype=np.float32)
			count = -1

			# load the training data
			ii = 0
			for i in range(len(subNum_tr)):
				for ii in range(len(tasks)):
					filename = 'P' + str(subNum_tr[i]) + 'T' + str(tasks[ii]) + '.mat'
					s = os.path.join(data_path_blue, filename)
					print(filename)
					# print(s)
					data = h5py.File(s)
					noise_blue = np.array(data["mask_noise_blue_mean2"])
					# print(data.keys()
					#data = scipy.io.loadmat(s)
						
					s = os.path.join(data_path_green, filename)
					data = h5py.File(s)
					noise_green = np.array(data["mask_noise_green_mean2"])

					s = os.path.join(data_path_red, filename)
					data = h5py.File(s)
					noise_red = np.array(data["mask_noise_red_mean2"])

					noisy_signal = np.array(data["yptest_sub2"])
					gt_signal = np.array(data["pulse_ox"]) # here the variable pulse_ox is the respiration signal
							
					# remove the last time point which is 0
					noise_red = noise_red[:,0:36000]
					noise_green = noise_green[:,0:36000]
					noise_blue = noise_blue[:,0:36000]
					noisy_signal = noisy_signal[:,0:36000]
					gt_signal = gt_signal[:,0:36000]
					# downsample to 30 fps from 120
					noise_red = noise_red[:,::4]
					noise_red = np.transpose(noise_red)
					noise_green = noise_green[:,::4]
					noise_green = np.transpose(noise_green)
					noise_blue = noise_blue[:,::4]
					noise_blue = np.transpose(noise_blue)
					noisy_signal = noisy_signal[:,::4]
					noisy_signal = np.transpose(noisy_signal)
					gt_signal = gt_signal[::4,:] # don't transpose gt_signal, it has different shape

					# normalize to -1 to 1 - do this in a better way for the whole dataset
					noise_red = noise_red / max(abs(noise_red))
					noise_green = noise_green / max(abs(noise_green))
					noise_blue = noise_blue / max(abs(noise_blue))
					noisy_signal = noisy_signal / max(abs(noisy_signal))
					gt_signal = gt_signal / max(abs(gt_signal))
						
					for j in range(0,len(noisy_signal) - (n_examples-1), int(n_examples/2)):						
						count = count+1

						Xtrain1[count:(count+1), 0:n_examples] = np.transpose(noise_red[j:j+n_examples,:])
						Xtrain2[count:(count+1), 0:n_examples] = np.transpose(noise_green[j:j+n_examples,:])
						Xtrain3[count:(count+1), 0:n_examples] = np.transpose(noise_blue[j:j+n_examples,:])
						Xtrain4[count:(count+1), 0:n_examples] = np.transpose(noisy_signal[j:j+n_examples,:])
						Ytrain[count:(count+1), 0:n_examples] = np.transpose(gt_signal[j:j+n_examples,:])   
			# load the test data
			count = -1
			ii = 0			
			print('Test Set files')
			for i in range(len(subNum_ts)):
				for ii in range(len(tasks)):
					filename = 'P' + str(subNum_ts[i]) + 'T' + str(tasks[ii]) + '.mat'
					s = os.path.join(data_path_blue, filename)
					print(filename)
					data = h5py.File(s)
					noise_blue = np.array(data["mask_noise_blue_mean2"])
					
					s = os.path.join(data_path_green, filename)
					data = h5py.File(s)
					noise_green = np.array(data["mask_noise_green_mean2"])

					s = os.path.join(data_path_red, filename)
					data = h5py.File(s)
					noise_red = np.array(data["mask_noise_red_mean2"])

					noisy_signal = np.array(data["yptest_sub2"])
					gt_signal = np.array(data["pulse_ox"]) # here the variable pulse_ox is the respiration signal
							
					# remove the last time point which is 0
					noise_red = noise_red[:,0:36000]
					noise_green = noise_green[:,0:36000]
					noise_blue = noise_blue[:,0:36000]
					noisy_signal = noisy_signal[:,0:36000]
					gt_signal = gt_signal[:,0:36000]
					# downsample to 30 fps from 120
					noise_red = noise_red[:,::4]
					noise_red = np.transpose(noise_red)
					noise_green = noise_green[:,::4]
					noise_green = np.transpose(noise_green)
					noise_blue = noise_blue[:,::4]
					noise_blue = np.transpose(noise_blue)
					noisy_signal = noisy_signal[:,::4]
						
					noisy_signal = np.transpose(noisy_signal)
					gt_signal = gt_signal[::4,:] # don't transpose gt_signal, it has different shape

					# normalize to -1 to 1 - do this in a better way for the whole dataset
					noise_red = noise_red / max(abs(noise_red))
					noise_green = noise_green / max(abs(noise_green))
					noise_blue = noise_blue / max(abs(noise_blue))
					noisy_signal = noisy_signal / max(abs(noisy_signal))
					gt_signal = gt_signal / max(abs(gt_signal))

					# split to short sequences	
					for j in range(0,len(noisy_signal) - (n_examples-1), int(n_examples/2)):
						count = count+1
						Xtest1[count:(count+1), 0:n_examples] = np.transpose(noise_red[j:j+n_examples,:])
						Xtest2[count:(count+1), 0:n_examples] = np.transpose(noise_green[j:j+n_examples,:])
						Xtest3[count:(count+1), 0:n_examples] = np.transpose(noise_blue[j:j+n_examples,:])
						Xtest4[count:(count+1), 0:n_examples] = np.transpose(noisy_signal[j:j+n_examples,:])
						Ytest[count:(count+1), 0:n_examples] = np.transpose(gt_signal[j:j+n_examples,:])  

			Xtrain[:,:,0] = Xtrain1
			Xtrain[:,:,1] = Xtrain2
			Xtrain[:,:,2] = Xtrain3
			Xtrain[:,:,3] = Xtrain4
			Xtest[:,:,0] = Xtest1
			Xtest[:,:,1] = Xtest2
			Xtest[:,:,2] = Xtest3
			Xtest[:,:,3] = Xtest4

			Ytrain = Ytrain.reshape(n_samples_tr, n_examples, 1)
			Ytest = Ytest.reshape(n_samples_ts, n_examples, 1)

			# save data
			if os.path.exists(save_path + 'task' + str(task) + '/') == 0:
				os.mkdir(save_path + 'task' + str(task) + '/')
			if os.path.exists(save_path + 'task' + str(task) + '/split' + str(split) + '/') == 0:
				os.mkdir(save_path + 'task' + str(task) + '/split' + str(split) + '/')
			scipy.io.savemat(save_path + 'task' + str(task) + '/split' + str(split)  + '/Xtrain.mat', mdict={'Xtrain': Xtrain})
			scipy.io.savemat(save_path + 'task' + str(task) + '/split' + str(split)  + '/Ytrain.mat', mdict={'Ytrain': Ytrain})
			scipy.io.savemat(save_path + 'task' + str(task) + '/split' + str(split)  + '/Xtest.mat', mdict={'Xtest': Xtest})
			scipy.io.savemat(save_path + 'task' + str(task) + '/split' + str(split)  + '/Ytest.mat', mdict={'Ytest': Ytest})

			# loop over different hidden dimension parameters, if testing different architectures, otherwise just use one value
			for h in range(len(hidden_dim_list)):
				loss_tr_save = []
				loss_ts_save = []
				epoch_save = []
				hidden_dim = hidden_dim_list[h]
				print('processing # hidden dimensions ' + str(hidden_dim))

				# create LSTM
				model = Sequential()
				model.add(LSTM(hidden_dim, input_shape=(n_examples,n_numbers)))
				model.add(RepeatVector(n_examples))
				model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True)))

				model.add(TimeDistributed(Dense(n_chars, activation='tanh')))
				model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

				print(model.summary())
				# train LSTM
				rmse_tr = list()
				for epoch in range(n_epoch):
					model.fit(Xtrain, Ytrain, epochs=1, batch_size=n_batch, verbose=0)
					result_tr = model.predict(Xtrain, verbose=0)
					expected_tr = Ytrain
					predicted_tr = result_tr
					input_tr = Xtrain

					rmse_tr_i = sqrt(mean_squared_error(expected_tr[:,:,0], predicted_tr[:,:,0]))
					rmse_tr.append(rmse_tr_i)
					loss_tr = mean_squared_error(expected_tr[:,:,0], predicted_tr[:,:,0])
					loss_tr_save.append(loss_tr)
					epoch_save.append(epoch)

					print('epoch #: ' + str(epoch))
					
					# evaluate on some new patterns
					result = model.predict(Xtest, verbose=0)
					expected = Ytest
					predicted = result

					expected = np.array(expected)
					predicted = np.array(predicted)

					loss_ts = mean_squared_error(expected[:,:,0], predicted[:,:,0])
					loss_ts_save.append(loss_ts)
					print('test loss')
					print(loss_ts)

					if epoch % 5 == 0:
						print('saving epoch #: ' + str(epoch))
						# save predicted values
						scipy.io.savemat(save_path + '/task' + str(task) + '/split' + str(split) + '/predicted' + str(hidden_dim) +  str(epoch) + '.mat', mdict={'predicted': predicted})
						scipy.io.savemat(save_path + '/task' + str(task) + '/split' + str(split) + '/expected' + str(hidden_dim) +  str(epoch) + '.mat', mdict={'expected': expected})
						scipy.io.savemat(save_path + '/task' + str(task) + '/split' + str(split) + '/predicted_tr' + str(hidden_dim) +  str(epoch) + '.mat', mdict={'predicted_tr': predicted_tr})
						scipy.io.savemat(save_path + '/task' + str(task) + '/split' + str(split) + '/expected_tr' + str(hidden_dim) +  str(epoch) + '.mat', mdict={'expected_tr': expected_tr})
						
						scipy.io.savemat(save_path + '/task' + str(task) + '/split' + str(split) + '/loss_tr' + str(hidden_dim) +  str(epoch) + '.mat', mdict={'loss_tr': loss_tr})
						scipy.io.savemat(save_path + '/task' + str(task) + '/split' + str(split) + '/loss_ts' + str(hidden_dim) +  str(epoch) + '.mat', mdict={'loss_ts': loss_ts})
						
						## save model 
						model.save(save_path + '/task' + str(task) + '/split' + str(split) + '/model' + str(hidden_dim) +  str(epoch) + '.h5')
							
						plt.plot(epoch_save, loss_tr_save)
						plt.plot(epoch_save, loss_ts_save)
						plt.title('model loss')
						plt.ylabel('loss')
						plt.xlabel('epoch')
						plt.legend(['train', 'test'], loc='upper left')
						plt.savefig(save_path + '/task' + str(task) + '/split' + str(split) + '/loss_vs_epoch' + str(hidden_dim) +  str(epoch) + '.png')
						plt.close()