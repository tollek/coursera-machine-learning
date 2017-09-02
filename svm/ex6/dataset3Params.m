function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;
return

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

cand = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
# cartesian product: cand x cand x 0:0
# we will update the params and then find one with lowest error
# copy-pasted implementation of cartprod
[params{1:3}] = ndgrid(cand,cand,0:0);
params = cat(4, params{:});
params = reshape(params, [], 3);

%params(6

for i = 1:length(params)
  C_cand = params(i, 1);
  sigma_cand = params(i, 2);
  
  # train model
  model= svmTrain(X, y, C_cand, @(x1, x2) gaussianKernel(x1, x2, sigma_cand)); 
  pred = svmPredict(model, Xval);
  err = mean(double(pred ~= yval));
  params(i,3) = err;
end

params

[_, best_param] = min(params(:,3));
C = params(best_param, 1);
sigma = params(best_param, 2);

% best results:
% c =  1
% sigma =  0.10000

% =========================================================================

end
