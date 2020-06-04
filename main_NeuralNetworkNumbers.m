%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % ("0" mapped to label 10 for simplicity)

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Training data stored in arrays X, y through MATLAB
% X is a 5000x400 matrix containing 5000 examples of handwritten data
% stored as 400-element vectors unwrapped from 20x20 pixel grayscale images
% y contains labels ranging from 1 to 10
load('trainingData.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50);

%  Setting regularization parameter
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nVisualizing Neural Network... \n')
displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Predictions
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% Testing with manually written bitmaps (stored in five.bmp)
close all;

map = double(imread("five.bmp"));
% Roll out elements across single dimensional vector
for i = 1:20
	for j = 1:20
		mapvec(1, (j-1)*20 + i) = map(i,j);
	end
end

% Limiting range between 0 and 1
mapvec = mapvec/255;

% Displaying 400x1 vector on display
displayData(mapvec);

% Neural network prediction
nnPrediction = predict(Theta1,Theta2,mapvec);

% rem() is used to convert 10 to 0
fprintf('\nNeural Network Prediction: %d\n',rem(nnPrediction,10));
fprintf('\nPress any key to exit.\n');
pause;
close all;