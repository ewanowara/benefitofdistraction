vidPath; % please provide a video path after downloading a dataset
load('Data/sample_mask1.mat') 
mask_noise = zeros(size(mask1));

% bandpass filtering parameters
L = 34;
% for HR
fs = 30; % video frame rate
[b,a] = butter(1,[0.7/fs*2 2.5/fs*2]);

% for BR
% [b,a] = butter(1,[0.08/fs*2 0.5/fs*2]);

for t = 1:size(mask1,1)

    mask_tmp = squeeze(mask1(t,:,:));
    mask_tmp = imresize(mask_tmp, [34 34]);

    % normalize
    mask_tmp = mask_tmp - min(mask_tmp(:));
    mask_tmp = mask_tmp./max(mask_tmp(:));

    mask_noise_tmp = mask_tmp(:);
    thresh = 0.1;

    mask_noise_tmp(mask_tmp(:) > thresh) = 0;
    mask_noise_tmp(mask_tmp(:) <= thresh) = 1;

    mask_noise_tmp = reshape(mask_noise_tmp, [L L]);

    mask_noise(t,:,:) = mask_noise_tmp;
end
save('Data/sample_mask_noise_new.mat',  'mask_noise', '-v7.3') 

%% use the mask_noise to create the red, green and channel noise estimates
load('Data/sample_yptest.mat') % predicted BR or HR signal

for RGB_channel = 1:3
    mask_noise_i_tmp = mask_noise;

    % load video frames
    mask_noise_channel = zeros(size(yptest,1),L, L);
    v = dir([vidPath '*.jpg']);
    j = 0;
    for i = 1:length(v)
        j=j+1;
        vidFrame = imread([vidPath v(i).name]);
%         Width = size(vidFrame,2);
%         Height = size(vidFrame,1);

%         vidLxL = imresize(im2single(vidFrame(:,Width/2-Height/2+1:Width/2+Height/2,:)),[L,L]);
                
        vidLxL = imresize(im2single(vidFrame),[L,L]);
        vidLxL(vidLxL>1) = 1;
        vidLxL(vidLxL<1/255) = 1/255; 
        % multiply mask by image
        mask_noise_channel(j,:,:) = vidLxL(:,:,RGB_channel) .* squeeze(mask_noise_i_tmp(j,:,:));
    end

    mask_noise_channel_mean = mean(mean(mask_noise_channel,3),2);
    yptest_sub1 = cumsum(yptest);

    lambda=50;
    T=length(yptest_sub1);
    I=speye(T);
    D2=spdiags(ones(T-2,1)*[1 -2 1],[0:2],T-2,T);
    sr=double(yptest_sub1);

    temp = I*sr - (sr'/ (I+lambda^2*D2'*D2))';
    nZ = (temp - mean(temp))/std(temp);

    yptest_sub2 = filtfilt(b,a,double(nZ));

    T=length(mask_noise_channel_mean);
    I=speye(T);
    D2=spdiags(ones(T-2,1)*[1 -2 1],[0:2],T-2,T);
    sr=double(mask_noise_channel_mean);
    temp = I*sr - (sr'/ (I+lambda^2*D2'*D2))';
    nZ2 = (temp - mean(temp))/std(temp);

    mask_noise_channel_mean2 = filtfilt(b,a,double(nZ2));

    % rename variables to red, green, channel:
    if RGB_channel == 1
        mask_noise_red = mask_noise_channel;
        mask_noise_red_mean = mask_noise_channel_mean;
        mask_noise_red_mean2 = mask_noise_channel_mean2;
        save('Data/RGB_noise_estimates_red/sample_noise_new.mat', 'yptest', 'yptest_sub2', 'mask_noise_red', 'mask_noise_red_mean', 'mask_noise_red_mean2', '-v7.3');

    elseif RGB_channel == 2
        mask_noise_green = mask_noise_channel;
        mask_noise_green_mean = mask_noise_channel_mean;
        mask_noise_green_mean2 = mask_noise_channel_mean2;
        save('Data/RGB_noise_estimates_green/sample_noise_new.mat', 'yptest', 'yptest_sub2', 'mask_noise_green', 'mask_noise_green_mean', 'mask_noise_green_mean2', '-v7.3');

    elseif RGB_channel == 3
        mask_noise_blue = mask_noise_channel;
        mask_noise_green_mean = mask_noise_channel_mean;
        mask_noise_green_mean2 = mask_noise_channel_mean2;
        save('Data/RGB_noise_estimates_blue/sample_noise_new.mat', 'yptest', 'yptest_sub2', 'mask_noise_blue', 'mask_noise_blue_mean', 'mask_noise_blue_mean2', '-v7.3');
    end    
end