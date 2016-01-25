function [prob, comp_probs] = cwmProb(ratings, features, feature_means, feature_covars, feature_weights, feature_biases, rating_vars, cluster_priors)

% Estimate conditional probabilities of ratings given features using a 
% trained CWM model. Usually called by emFitCWM.
% Parameters:
%	ratings - the ratings associated with each example
%	features - the feature values for each example (organized in a 
%		matrix - columns are feature dimensions and rows are examples)
%	feature_means - CWM cluster means
%	feature_covars - CWM cluster covariances
%	feature_weights - CWM cluster regression weights
%	feature_biases - CWM cluster regression biases
%	rating_vars - CWM cluster rating variances
%	cluster_priors - CWM cluster mixture proportions
% Return Values:
%	prob - estimated conditional probabilities of the ratings
%	comp_probs - the contribution of each cluster
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
dim = size(features, 2);

for k = 1:num_clusters
	zero_mean_data = features - repmat(feature_means(k,:), num_samples, 1);
	g_features = exp(-0.5 * sum((zero_mean_data * inv(feature_covars(:,:,k))) .* zero_mean_data, 2)) ./ ((2 * pi)^(dim / 2) * det(feature_covars(:,:,k))^(1 / 2));
	g_rating = exp(-0.5 * (ratings - feature_biases(k) - features * feature_weights(k,:)').^2 ./ rating_vars(k)) ./ (sqrt(2 * pi * rating_vars(k)));
	comp_probs(:,k) = g_features .* g_rating .* cluster_priors(k);
end

prob = sum(comp_probs, 2);

return;