

load('result.mat');

folder = fullfile('/Users/admin/Desktop/');
baseFileName = 'xxx.jpg';
fullFileName = fullfile(folder, baseFileName);
rgbImage = imread(fullFileName);

[featureVector, hogVisualization] = extractHOGFeatures(imresize(rgbImage,[160,160]),'CellSize',[16 16]); 
Features = featureVector*COEFF (:,1:24);
rgbImage = imresize(rgbImage,[160,160]);
Features = [colorMoments(rgbImage)./10 lbp(rgbImage)*COEFF2(:,1:8)./1000 Features];
[ratings1, mix_probs1, expected_vars1] = cwmEstimate(Features, feature_means, feature_covars, feature_weights, feature_biases, rating_vars, cluster_priors);
