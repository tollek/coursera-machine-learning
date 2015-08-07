function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X * theta);
theta_no0 = theta([2:size(theta,1)], :); % theta[2:, :] - with theta0 removed (don't regularize it)

J = 1 / m * sum(-y .* log(h) - (1 - y) .* log(1-h)) +  lambda / (2 * m) * sum(theta_no0 .^2);

for j = 1:length(theta)
    reg = 0;
    if j > 1
        reg = lambda / m * theta(j, 1);
    endif

    temp = 1 / m * sum((h - y) .* X(:, j)) + reg;
    grad(j, 1) = temp;





% =============================================================

end
