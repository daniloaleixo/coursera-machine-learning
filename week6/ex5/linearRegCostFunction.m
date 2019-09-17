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
%               You should set J to thes cost and grad to the gradient.
%
% X = [ ones(size(X)) X];
% lambda = 1


reg = lambda * (theta(2:end)' * theta(2:end));
J = ((X * theta - y)' * (X * theta - y) + reg) / 2 / m;

aux = X' * (X * theta - y);
reg2 = lambda .* [ 0; theta(2:end)];
grad = (aux + reg2) ./ m;











% =========================================================================

grad = grad(:);

end
