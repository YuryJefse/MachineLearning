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
sigma = 0.3;

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

possibles = [0.01 0.03 0.1 0.3 1 3 10 30];

result = zeros(64, 3);
indice = 0;

for i = 1 : length(possibles)
  for j = 1 : length(possibles) 
    indice = indice + 1;
    model = svmTrain(X, y, possibles(i), @(x1,x2) gaussianKernel(x1, x2, possibles(j)));
    predic = svmPredict(model, Xval);
    error = mean(double(predic ~= yval));
    
    result(indice, :) = [possibles(i), possibles(j), error];
  endfor
endfor

linha = sortrows(result, 3)(1 , : );

C = linha(1);
sigma = linha(2);

% =========================================================================

end
