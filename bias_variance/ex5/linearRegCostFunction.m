function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

H = X * theta;
J = 1/(2*m) * sum((H - y).^2) + lambda/(2*m) * sum(theta(2:end).^2);

% =========================================================================
% gradient:
% dJ/dTheta_j = 1/m * (sum_1_m (h(x_i)-y_i) * x_j_i) + lambda/m * theta_j^2
ghelper = repmat(H-y, 1, size(X,2));
non_reg_grad = 1/m * (ghelper .* X);
non_reg_grad = sum(non_reg_grad);
% add regularaziation for theta_j (j>0)
% gradient == derivative of J(Theta), thus the regularization therm is Theta, not Theta^2
grad = non_reg_grad + lambda/m*[0, theta(2:end)'];

grad = grad(:);

end

