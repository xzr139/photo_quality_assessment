function [feature_means, feature_covars, feature_weights, feature_biases, rating_vars, cluster_priors, log_likelihood] = emFitCWM(ratings, features, num_clusters)

% Fit a CWM model for predicting the ratings from the features.
% Parameters:
%	ratings - the ratings associated with each example
%	features - the feature values for each example (organized in a 
%		matrix - columns are feature dimensions and rows are examples)
%	num_clusters - the number of regression clusters to include in the model
% Return Values:
%	feature_means - CWM cluster means
%	feature_covars - CWM cluster covariances
%	feature_weights - CWM cluster regression weights
%	feature_biases - CWM cluster regression biases
%	rating_vars - CWM cluster rating variances
%	cluster_priors - CWM cluster mixture proportions
%	log_likelihood - log likelihood of the model parameters
% Based on Gershenfeld (1999) "The Nature of Mathematical Modeling"
%
% Michael Ross
% mgross@alum.mit.edu
%
% This code was developed and is made available solely for educational,
% academic, and research purposes. It was used to generate results presented
% in "Estimating perception of scene layout properties from global image
% features" by Michael G. Ross and Aude Oliva, published in the Journal of
% Vision (2010).

num_samples = size(features, 1);
feat_dim = size(features, 2);
rand_sel = randperm(num_samples);

feature_means = features(rand_sel(1:num_clusters),:);
var_feats = var(features);
for k = 1:num_clusters
	feature_covars(:,:,k) = diag(var_feats) + eye(feat_dim) * 0.00001;
end

feature_weights = zeros(num_clusters, feat_dim);
feature_biases = ratings(rand_sel(1:num_clusters),:);
rating_vars = var(ratings) * ones(num_clusters, 1) + 0.00001;
cluster_priors = ones(num_clusters, 1) / num_clusters;

rep_ratings = repmat(ratings, 1, feat_dim);

abs_delta_param = inf;
log_likelihood = -inf;

while abs_delta_param > 1e-8
	[probs, comp_probs] = cwmProb(ratings, features, feature_means, feature_covars, feature_weights, feature_biases, rating_vars, cluster_priors);
	old_log_likelihood = log_likelihood;
	log_likelihood = sum(log(probs));
	
	%disp(sprintf('log likelihood=%f |delta params|=%f', log_likelihood, abs_delta_param));
	
	old_feature_means = feature_means;
	old_feature_covars = feature_covars;
	old_feature_weights = feature_weights;
	old_feature_biases = feature_biases;
	old_rating_vars = rating_vars;
	old_cluster_priors = cluster_priors;
	
	resp = comp_probs ./ repmat(probs, 1, num_clusters);
	resp_sums = sum(resp, 1);
	
	for k = 1:num_clusters
		rep_resp = repmat(resp(:,k), 1, feat_dim);
		feature_means(k,:) = sum(features .* rep_resp, 1) / resp_sums(k);
		zero_mean_data = features - repmat(feature_means(k,:), num_samples, 1);
		feature_covars(:,:,k) = ((rep_resp .* zero_mean_data)' * zero_mean_data) / resp_sums(k) + eye(feat_dim) * 0.00001;
		feature_weights_ext = inv([[feature_covars(:,:,k); zeros(1, feat_dim)] zeros(feat_dim + 1, 1)] + [feature_means(k,:)'; 1] * [feature_means(k,:) 1]) * ([sum(rep_ratings .* features .* rep_resp, 1) sum(ratings .* resp(:,k), 1)] / resp_sums(k))';
		feature_weights(k,:) = feature_weights_ext(1:(end - 1));
		feature_biases(k) = feature_weights_ext(end);
		rating_vars(k) = sum((ratings - feature_biases(k) - features * feature_weights(k,:)').^2 .* resp(:,k), 1) / resp_sums(k) + 0.00001;
	end
	
	cluster_priors = resp_sums / sum(resp_sums);
	
	abs_delta_param = sum(abs(old_feature_means(:) - feature_means(:))) + sum(abs(old_feature_covars(:) - feature_covars(:))) + sum(abs(old_feature_weights(:) - feature_weights(:))) + sum(abs(old_feature_biases(:) - feature_biases(:))) + sum(abs(old_rating_vars(:) - rating_vars(:))) + sum(abs(old_cluster_priors(:) - cluster_priors(:)));
end

[probs, comp_probs] = cwmProb(ratings, features, feature_means, feature_covars, feature_weights, feature_biases, rating_vars, cluster_priors);
log_likelihood = sum(log(probs));

%disp(sprintf('log likelihood: %f', log_likelihood));

return;