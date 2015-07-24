function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Using the gradientDescent, as it's written general enough to handle multivariable scenario.
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

end

