function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Forward propagation
a1 = [ones(size(X, 1), 1), X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1), a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% cost function - iterative (over each sample) approach
for i = 1:m
	% apppends '1' as first column
	y_delta = zeros(num_labels, 1);
	y_delta(y(i)) = 1;

	a3_delta = a3(i, :)';
	% j_delta is the sum_k [..] in the J(theta) formula
	j_delta = -y_delta.*log(a3_delta) - (1-y_delta).*log(1-a3_delta);
	J = J + sum(j_delta);
end

% cost function - vectorized approach
%	y_matrix = zeros(size(y,1), num_labels);
%	for i = 1:m
%		y_matrix(i, y(i)) = 1;
%	end
%	J = sum(sum(-y_matrix .* log(a3) - (1-y_matrix) .* log(1-a3)));
	
J = J / m;

% add regularization
reg = 0;
reg = reg + sum(sum(Theta1(:, 2:end) .^ 2));
reg = reg + sum(sum(Theta2(:, 2:end) .^ 2));
reg = lambda / (2 * m) * reg;
J = J + reg;


% -------------------------------------------------------------
% Compute gradients
for i = 1:m
	% step 1: calculate input layer's values
	% DONE

	% step 2: calculate d_k for k = 3 (output layer)
	% d_k3 = a_k3 - y_k
	% we set values of a_k3 for every value, except the y(i) - here "-1"
	d_k3 = a3(i, :) - ([1:num_labels]==y(i));
	d_k3 = d_k3';	% d_k3 is N x 1 vector

	% step 3:
	% d_2 = Theta2' * d_3 .* sigmoid(z_2)
	% we remove Theta2_0 (bias element)
	% we transpose z2(i, :) - we want column vector
	% NOTE: this is equivalent to removing delta2_0
	d_k2 = (Theta2(:,2:end)' * d_k3) .* sigmoidGradient(z2(i, :)');

	% step 4:
	% DELTA(l) = DELTA(l) + d_(l+1) * a_l'		
	Theta2_grad = Theta2_grad + d_k3 * a2(i,:);
	Theta1_grad = Theta1_grad + d_k2 * a1(i,:);

end

% Gradient for every Theta is 1/m * DELTA_ij
% second component comes from regularization
Theta2_grad = Theta2_grad / m + [zeros(size(Theta2,1),1), Theta2(:, 2:end)] * lambda / m;
Theta1_grad = Theta1_grad / m + [zeros(size(Theta1,1),1), Theta1(:, 2:end)] * lambda / m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

