# The Benefit of Distraction: Denoising Camera-Based Physiological Measurements using Inverse Attention
We present a denoising method which uses regions ignored by attention networks as corruption estimates to denoise temporal signals of interest. We present results on the task of camera-based heart rate and breathing rate estimation. Please see this video for example results: https://www.dropbox.com/s/v7zfcnr2fuffgs1/video_Benefit_of_distraction.mp4?dl=0

Attention networks are commonly used in computer vision and they often perform strongly on diverse vision tasks. In computer vision, the core idea behind attention is that the signal of interest is stronger in some pixels ("foreground"), and by selectively focusing computation on these pixels, networks can extract subtle signals buried in noise and other sources of corruption. Our paper is based on one key observation: in many real-world applications, many sources of corruption, such as illumination and motion are often shared between the "foreground" and the "background" pixels. Can we utilize this to our advantage? We propose the utility of inverse attention networks, which focus on extracting information about these shared sources of corruption. We show that this allows us to build models that effectively suppress shared covariates and amplify signal information, resulting in improved performance. We illustrate this on the task of camera-based physiological measurement where the signal of interest is weak and global illumination variations and motion act as significant shared sources of corruption. We perform experiments on three datasets and show that our approach of inverse attention produces state-of-the-art results, increasing the signal-to-noise ratio by up to 5.8 dB, reducing heart rate and breathing rate estimation errors by as much as 30 %, recovering subtle waveform dynamics, and generalizing from RGB to NIR videos without retraining. 

## Supplementary Material: https://github.com/AnonymousCodeSubmission/Benefit_of_Distraction/blob/master/SupplementaryMaterial.pdf

## The architecture of our proposed denoising approach
<img src = Data/denoising_architecture8_corr.png>

## Example result
<img src = Data/Overview.png>

## Example result
<img src = Data/Masks_Examples4_corr.png>

## To reproduce our results:

Please download the MMSE-HR dataset (Z. Zhang, J. Girard, Y. Wu, X. Zhang, P. Liu, U. Ciftci, S. Canavan, M. Reale, A. Horowitz, H. Yang, J. F. Cohn, Q. Ji, and L. Yin. Multimodal spontaneous emotion corpus for human behavior analysis. In CVPR, 2016.) to train models from scratch.

A demo of this code will run on the included sample sequences from one video of the MMSE-HR dataset.

The code is currently implemented in several separate steps:

## 1. Estimate initial physiological signals and attention masks

Load a pre-trained convolutional attention network (CAN) model to extract heart rate (HR) or breathing rate (BR), output a pulse or respiration signal, and an attention mask showing which regions in an image were used to compute the physiological signals.

For HR:
```
$ python get_HR/get_initial_HR_load_CAN_model.py
```

For BR:
```
$ python get_BR/get_initial_BR_load_CAN_model.py
```

Or train a CAN model from scratch:
For HR:
```
$ python get_HR/Training/get_initial_HR_train_CAN_model.py
```

For BR:
```
$ python get_BR/Training/get_initial_BR_train_CAN_model.py
```

## 2. Compute corruption estimates:

MATLAB: 
```$ run get_noise_estimates.m```

It computes the inverse attention masks from the original attention masks and corruption estimates for each R,G,B camera channel by elementwise multiplying the inverse attention masks with the video frames.

## 3. Denoise the physiological signals using the noise estimates

Load a pre-trained LSTM model to denoise signals
For HR:
```
$ python get_HR/denoising_LSTM_load_model.py
```

For BR:
```
$ python get_BR/denoising_LSTM_load_model.py
```

Or train LSTM from scratch:
For HR:
```
$ python get_HR/Training/denoising_LSTM_train_model.py
```

For BR:
```
$ python get_BR/Training/denoising_LSTM_train_model.py
```

## 4. Print the results:

Compute the HR or BR and error metrics (MAE, RMSE, correlation coefficient, SNR, Wave MAE) from the denoised physiological signal. 

MATLAB: 
```$ run print_result.m```

## Data
The Data folder contains a short sequence of video frames as an example in example_video, along with the corresponding attention_mask (attention_mask.mat) and estimated pulse signal (BVP_estimate_from_CAN.mat) output by CAN. Example R, G, B noise estimates, estimated, and ground truth BVP signals are provided in Noise_masks_red_tr, Noise_masks_green_tr, Noise_masks_blue_tr folders for training and Noise_masks_red_ts, Noise_masks_green_ts, Noise_masks_blue_ts for testing. 

Please download the full MMSE-HR dataset or a different video dataset with ground truth physiological signals to train the model from scratch. 

