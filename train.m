clc    
clear
load('LandscapeData.mat')

s = landscape2(:,end);

r1 = find(s>=1);
r1 = r1(1:80,:);
r2 = find(s>=2);
r2 = r2(1:80,:);
r3 = find(s>=3);
r3 = r3(1:80,:);
r4 = find(s>=4);
r4 = r4(1:80,:);
r5 = find(s>=1);
r5 = r5(end-79:end,:);
r = [r1;r2;r3;r4;r5];

ids = landscape2(r, 2);

r = setdiff(find(s>=1),r);
s(r,:) = []; 

%HOG

numDests = numel(ids);
numFeatures = 2916;
Features1 = zeros(numDests, numFeatures);

for i = 1:size(s)
    img = imread(['/Users/admin/Desktop/imageability/photo/' num2str(ids(i)) '.jpg']);
    [featureVector, hogVisualization] = extractHOGFeatures(imresize(img,[160,160]),'CellSize',[16 16]); 
    Features1(i, :) = featureVector;
end

[COEFF,SCORE,latent,tsquare] = princomp(Features1);
Features1 = Features1 * COEFF(:,1:24);

%Monments

numDests = numel(ids);
numFeatures = 6;
Features3 = zeros(numDests, numFeatures);

for i = 1:size(s)
    img = imread(['/Users/admin/Desktop/imageability/photo/' num2str(ids(i)) '.jpg']);
    rgbImage = imresize(img,[160,160]);
    Features3(i, :) = colorMoments(rgbImage)./10;
end

%LBP

numDests = numel(ids);
numFeatures = 256;
Features2 = zeros(numDests, numFeatures);

for i = 1:size(s)
    img = imread(['/Users/admin/Desktop/imageability/photo/' num2str(ids(i)) '.jpg']);
    rgbImage = imresize(img,[160,160]);
    Features2(i, :) = lbp(rgbImage);
end

[COEFF2,SCORE,latent,tsquare] = princomp(Features2);
Features2 = Features2 * COEFF2(:,1:8)./1000;

%%%%%%%%

Features = [Features3 Features2 Features1];


[feature_means, feature_covars, feature_weights, feature_biases, rating_vars, cluster_priors, log_likelihood] = emFitCWM(s, Features, 4);
save result.mat COEFF COEFF2 feature_means feature_covars feature_weights feature_biases rating_vars cluster_priors log_likelihood







% =============================?????gist=============================

% 
% % Get Image ids
% ids = landscape2(:, 2);
% 
% % Specify GIST Parameters
% param.imageSize = [256 256]; % set a normalized image size
% param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
% param.numberBlocks = 4;
% param.fc_prefilt = 4;
% 
% % Preallocate GIST
% numFeatures = sum(param.orientationsPerScale)*param.numberBlocks^2;
% numDests = numel(ids);
% destGIST = zeros(numDests, numFeatures);
% 
% for i = 1:size(landscape2,1)
%     destGIST(i, :) = LMgist(imread(['/Users/admin/Desktop/imageability/photo/' num2str(ids(i)) '.jpg']), '', param);
% end
% 
% save MyGIST.mat destGIST

% load('MyGIST.mat')

% size(find(landscape2(:,end)==1))
% size(find(landscape2(:,end)==2))
% size(find(landscape2(:,end)==3))
% size(find(landscape2(:,end)==4))
% size(find(landscape2(:,end)==5))
% size(find(landscape2(:,end)==6))
% size(find(landscape2(:,end)==7))
% size(find(landscape2(:,end)==8))
% size(find(landscape2(:,end)==9))
% size(find(landscape2(:,end)==10))

% landscape2(1:575,end)=1;
% landscape2(576:1937,end)=2;
% landscape2(1938:3162,end)=3;
% landscape2(3163:4434,end)=4;
% landscape2(4435:4980,end)=5;
% 
% temp(1:500,:)=landscape2(1:500,:);
% temp2(1:500,:)=destGIST(1:500,:);
% temp(501:1000,:)=landscape2(576:1075,:);
% temp2(501:1000,:)=destGIST(576:1075,:);
% temp(1001:1500,:)=landscape2(1938:2437,:);
% temp2(1001:1500,:)=destGIST(1938:2437,:);
% temp(1501:2000,:)=landscape2(3163:3662,:);
% temp2(1501:2000,:)=destGIST(3163:3662,:);
% temp(2001:2500,:)=landscape2(4435:4934,:);
% temp2(2001:2500,:)=destGIST(4435:4934,:);

% for i = 0:9
%     a = i*498+1;
%     b = i*498+498;
%     landscape2(a:b,end)=ceil((i+1)/2);
% end
% 
% FC = fitcecoc(destGIST(:, :), landscape2(:,end));
% 

% [COEFF,SCORE,latent,tsquare] = princomp(destGIST);
% 
% save MyGIST.mat destGIST COEFF SCORE latent tsquare


% [feature_means, feature_covars, feature_weights, feature_biases, rating_vars, cluster_priors, log_likelihood] = emFitCWM(s, t, 4);
% 
% 
% folder = fullfile('/Users/admin/Desktop/photo/');
% baseFileName = 'bad.jpg';
% fullFileName = fullfile(folder, baseFileName);
% rgbImage = imread(fullFileName);
% 
% param.imageSize = [256 256];
% param.orientationsPerScale = [8 8 8 8];
% param.numberBlocks = 4;
% param.fc_prefilt = 4;
% 
% test = LMgist(rgbImage, '', param);
% test = test * COEFF(:,40);
% 
% [ratings, mix_probs, expected_vars] = cwmEstimate(test, feature_means, feature_covars, feature_weights, feature_biases, rating_vars, cluster_priors)
% 
% % FC = fitcecoc(temp2(:,:), temp(:,end));
% % save multiclassfier.mat FC
% 
% 


