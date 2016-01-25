function [ratings, mix_probs, expected_vars] = cwmEstimate(features, feature_means, feature_covars, feature_weights, feature_biases, rating_vars, cluster_priors)

% Estimate ratings using a trained CWM model.
% Parameters:
%	features - the feature values to be used in estimation (organized in a 
%		matrix - columns are feature dimensions and rows are examples)
%	feature_means - CWM cluster means, output from emFitCWM
%	feature_covars - CWM cluster covariances, output from emFitCWM
%	feature_weights - CWM cluster regression weights, output from emFitCWM
%	feature_biases - CWM cluster regression biases, output from emFitCWM
%	rating_vars - CWM cluster rating variances, output from emFitCWM
%	cluster_priors - CWM cluster mixture proportions, output from emFitCWM
% Return Values:
%	ratings - estimated ratings for the examples input in features
%	mix_probs - the cluster mixture percentages used for each rating
%	expected_vars - a measure of estimate reliability
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


num_clusters = length(cluster_priors);
num_samples = size(features, 1);
ratings = zeros(num_samples, 1);
expected_vars = zeros(num_samples, 1);
sum_cond_prob = zeros(num_samples, 1);
dim = size(features, 2);

for k = 1:num_clusters
	zero_mean_data = features - repmat(feature_means(k,:), num_samples, 1);
	g_features = exp(-0.5 * sum((zero_mean_data * inv(feature_covars(:,:,k))) .* zero_mean_data, 2)) ./ ((2 * pi)^(dim / 2) * det(feature_covars(:,:,k))^(1 / 2));
	cond_prob(:,k) = g_features * cluster_priors(k);
	sum_cond_prob = sum_cond_prob + cond_prob(:,k);
	ratings = ratings + (feature_biases(k) + features * feature_weights(k,:)') .* cond_prob(:,k);
	expected_vars = expected_vars + rating_vars(k) .* cond_prob(:,k);
end

mix_probs = cond_prob ./ repmat(sum_cond_prob, 1, num_clusters);
ratings = ratings ./ sum_cond_prob;
expected_vars = expected_vars ./ sum_cond_prob;

return;