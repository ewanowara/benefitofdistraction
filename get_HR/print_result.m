l=60; % length of sequences used in LSTM
ECG_fs = 1000; % ground truth signal frame rate
fs=30;

% bandpass filter parameters
% if computing heart rate(HR) 
[b,a] = butter(3,[0.7/fs*2 2.5/fs*2]);

% breathing rate (BR)
% [b,a] = butter(3,[0.08/fs*2 0.5/fs*2]);

HR_noisy = [];
HR_GT = [];
HR_denoised_LSTM = [];
SNR_denoised_LSTM = [];
SNR_noisy = [];
wave_dif_noisy = [];
wave_dif_denoised_LSTM = [];

% loop through other subjects in the dataset

load('../Data/sample_input_data_with_groundtruth.mat', 'GT_HR_bp', 'bp_gt_resamp', 'GT_HR_resamp_bp')
% GT_HR_bp - blood pressure waveform, used to compute wave mean absolute error (WMAE)
% bp_gt_resamp - blood pressure waveform, resampled to 30 fps, used to establish length of the PPG signal
% GT_HR_resamp_bp - estimated HR

% loads ground truth average HR
hr_gt = GT_HR_bp; 

% load predicted LSTM output and Xtest saved from LSTM input
load('../Data/sample_LSTM_input_data.mat')
load('../Data/sample_predicted_HR.mat')

noisy_sig = zeros(size(GT_HR_resamp_bp));
predicted_sig = zeros(size(GT_HR_resamp_bp));

for jj = 1:size(Xtest,1)-1
    noisy_sig((jj-1)*l/2+1:l/2*(jj+1)) = Xtest(jj,:,4);
    predicted_sig((jj-1)*l/2+1:l/2*(jj+1)) = predicted(jj,:);
end
        
% bandpass filter
predicted_sig = filtfilt(b,a,double(predicted_sig));
noisy_sig = filtfilt(b,a,double(noisy_sig));
bp_gt_resamp = filtfilt(b,a,double(bp_gt_resamp)); 

% mean subtract
predicted_sig = predicted_sig - mean(predicted_sig);
noisy_sig = noisy_sig - mean(noisy_sig);
bp_gt_resamp = bp_gt_resamp - mean(bp_gt_resamp);

% normalize to -1 to +1
predicted_sig = predicted_sig./max(abs(predicted_sig));
noisy_sig = noisy_sig./max(abs(noisy_sig));
bp_gt_resamp = bp_gt_resamp./max(abs(bp_gt_resamp));

bp_gt_resamp = bp_gt_resamp';

% get HR or BR for each 30 second time window, without overlap
for j = 0:30:min((floor(length(noisy_sig)/fs) - 30), (floor(length(hr_gt)/ECG_fs) - 30))      
    tSpan = [j*fs+1:(j+30)*fs];
    tSpan2 = [j*ECG_fs+1:(j+30)*ECG_fs];

    [pxx_LSTM,f] = periodogram(predicted_sig(tSpan),[],4*30*fs,fs);
    [pxx,f] = periodogram(noisy_sig(tSpan),[],4*30*fs,fs);

    %% compute HR and SNR       
    fmask = (f >= 0.7)&(f <= 2.5);
    frange = f(fmask);

    fmask2 = (f >= 0.7)&(f <= 4);
    frange2 = f(fmask2);

    HR_denoised_LSTM = [HR_denoised_LSTM frange(argmax(pxx_LSTM(fmask),1))*60];
    HR_noisy = [HR_noisy frange(argmax(pxx(fmask),1))*60];
    HR_GT = [HR_GT median(hr_gt(tSpan2))];

    wave_dif_noisy = [wave_dif_noisy mean(abs(noisy_sig(tSpan) - bp_gt_resamp(tSpan)))];
    wave_dif_denoised_LSTM = [wave_dif_denoised_LSTM mean(abs(predicted_sig(tSpan) - bp_gt_resamp(tSpan)))];

    gtmask1 = (f >= ((HR_GT(end)/60)-0.1))&(f <= ((HR_GT(end)/60)+0.1)); 
    gtmask2 = (f >= ((HR_GT(end)/60)*2-0.1))&(f <= ((HR_GT(end)/60)*2+0.1));

    sPower_noisy = sum(pxx(gtmask1|gtmask2));
    allPower_noisy = sum(pxx(fmask2));
    SNR_noisy = [SNR_noisy mag2db(sPower_noisy/(allPower_noisy-sPower_noisy))];

    sPower_denoised_LSTM = sum(pxx_LSTM(gtmask1|gtmask2));
    allPower_LSTM = sum(pxx_LSTM(fmask2));
    SNR_denoised_LSTM = [SNR_denoised_LSTM mag2db(sPower_denoised_LSTM/(allPower_LSTM-sPower_denoised_LSTM))];                                

end
            
MAE_noisy_ts = mean(abs(HR_noisy(:)-HR_GT(:)));
MAE_denoised_LSTM_ts = mean(abs(HR_denoised_LSTM(:)-HR_GT(:)));  

RMSE_noisy_ts = sqrt(mean((HR_noisy-HR_GT).^2));
RMSE_denoised_LSTM_ts = sqrt(mean((HR_denoised_LSTM-HR_GT).^2));

mean_SNR_noisy_ts = mean(SNR_noisy(:));
mean_SNR_denoised_LSTM_ts = mean(SNR_denoised_LSTM(:));

rho_noisy_ts = corrcoef(HR_noisy, HR_GT);
rho_denoised_LSTM_ts = corrcoef(HR_denoised_LSTM, HR_GT);

mean_wave_dif_noisy_ts = mean(wave_dif_noisy);
mean_wave_dif_denoised_LSTM_ts = mean(wave_dif_denoised_LSTM);
     
         
disp(['Ours MAE:     ' num2str(round(mean(MAE_denoised_LSTM_ts),2)) ' RMSE: ' ...
    num2str(round(mean(RMSE_denoised_LSTM_ts),2)) ' SNR: ' ...
    num2str(round(mean(mean_SNR_denoised_LSTM_ts),2)) ' correlation coeff.: ' ...
    num2str(round(mean(rho_denoised_LSTM_ts(1)),2)) ' WMAE: ' ...
    num2str(round(mean(mean_wave_dif_denoised_LSTM_ts),2)) ])

disp(['Baseline MAE: ' num2str(round(mean(MAE_noisy_ts),2)) ' RMSE: ' ...
    num2str(round(mean(RMSE_noisy_ts),2)) ' SNR: ' ...
    num2str(round(mean(mean_SNR_noisy_ts),2)) ' correlation coeff.: ' ...
    num2str(round(mean(rho_noisy_ts(1)),2))  ' WMAE: ' ...
    num2str(round(mean(mean_wave_dif_noisy_ts),2)) ]) 
 