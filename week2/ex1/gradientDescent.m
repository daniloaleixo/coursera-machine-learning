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

    sum_arr = zeros(2);
    for i = 1:m
      sum_arr(1) = sum_arr(1) + (theta' * X(i, :)' - y(i)) * X(i, 1);
      sum_arr(2) = sum_arr(2) + (theta' * X(i, :)' - y(i)) * X(i, 2);
    endfor

    temp_theta(1) = theta(1) - alpha * sum_arr(1) / m;
    temp_theta(2) = theta(2) - alpha * sum_arr(2) / m;

    % temp_theta


    theta = temp_theta';


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
