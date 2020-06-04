function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTION(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network.

% Reshaping nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Feedforward + cost calculation

X = [ones(m,1) X];
h_x = zeros(m, num_labels);

for t = 1:m
	z2 = Theta1 * X(t,:)';	% 25x401 // 401x1 = 25x1
	a2 = sigmoid(z2);		% 25x1
	z3 = Theta2 * [1; a2];	% 10x26 // 26x1 = 10x1
	h_x(t,:) = sigmoid(z3)';
end

% VECTORIZED IMPLEMENTATION - COST FUNCTION
for k = 1:num_labels
	yk = (y == k);
	J = J + (1/m)*(-(yk' * log(h_x(:,k))) - ((1 - yk)' * log(1 - h_x(:,k))));
end

% Regularized Cost Function
J = J + (lambda / (2*m)) * (sum(sum(Theta1(:,[2:end]).^2)) + sum(sum(Theta2(:,[2:end]).^2)));

% Backprop
delta_1 = zeros(hidden_layer_size,input_layer_size + 1);
delta_2 = zeros(num_labels,hidden_layer_size + 1);

for t = 1:m
	% Step 1 - Forward prop
	z2 = Theta1 * X(t,:)';	% 25x401 // 401x1 = 25x1
	a2 = sigmoid(z2);		% 25x1
	a2 = [1; a2];			% 26x1
	z3 = Theta2 * a2;	% 10x26 // 26x1 = 10x1
	a3 = sigmoid(z3);

	% Step 2 - Error calculation
	yVec = zeros(num_labels,1);
	% fprintf('----- Size of %s: %d x %d\n', 'yVec', size(yVec,1), size(yVec,2));
	yVec(y(t)) = 1;

	doh_3 = a3 - yVec; % 10x1
	doh_2 = (Theta2(:,[2:end])' * doh_3) .* sigmoidGradient(z2); 		% 25x10//10x1 -> 25x1 .* 25x1
	delta_2 = delta_2 + doh_3 * a2';		% 10x26 + 10x1//1x26
	delta_1 = delta_1 + doh_2 * X(t,:);
end

Theta1_grad = (1/m) * delta_1;
Theta1_grad = Theta1_grad + (lambda/m)*[zeros(hidden_layer_size,1) Theta1(:,[2:end])];
Theta2_grad = (1/m) * delta_2;
Theta2_grad = Theta2_grad + (lambda/m)*[zeros(num_labels,1) Theta2(:,[2:end])];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
