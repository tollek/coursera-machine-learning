function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %      
    y_pred = X * theta;
    # X .* (y_pred - y)        -- x(i,j) is individual contribution of sample i
    #                             into the change/gradient of j-th parameter
    # sum(X .* (y_pred - y))'  -- summarizes the individual contributions
    #                             and tranposes the vector to have same size as theta
    gradient = alpha * (1 / m) * sum(X .* (y_pred - y))';
    theta = theta - gradient;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    % print the cost
    % J_history(iter)
end

end
