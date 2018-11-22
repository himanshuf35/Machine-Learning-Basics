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
p = zeros(size(X, 1), 1);
X = [ones(m,1) , X];
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

z1 = X * Theta1';
a1 = sigmoid(z1);

a1 = [ones(m,1),a1];
z2 = a1 * Theta2';
h = sigmoid(z2);

for c = 1:num_labels,
J = J + ( (1/m) * ( ( -(y==c)' * log(h(:,c))) - ( (1-( y==c ))' * log(1-(h(:,c)) ) ) )  ) 
end;

temp1 = Theta1';
temp2 = Theta2';

Theta1_sum = sum( (temp1( 2:(input_layer_size + 1),: ) .^ 2)(:) );
Theta2_sum = sum( (temp2( 2:(hidden_layer_size + 1),: ) .^ 2)(:) );

J = J + ( ( lambda/( 2 * m ) ) * ( Theta1_sum + Theta2_sum ) );

Delta3 = zeros(num_labels,m);
for c1 =1:num_labels,
Delta3(c1,:) = h(:,c1) - (y==c1);
end;

Delta2 = zeros(hidden_layer_size,m);
Delta2 = (Theta2' * Delta3) .* (a1') .* ( 1-a1'); 
Delta2 = Delta2(2:(hidden_layer_size + 1),:);

GDelta2 = Delta3 * a1;
GDelta1 = Delta2 * X;

x = size(Theta2,2);
y = size(Theta1,2);
c = size(Theta2,2);

Theta1_grad(:,1) = (1/m) * GDelta1(:,1);
Theta1_grad(:,2:y) = ( (1/m) * ( GDelta1(:,2:y) ) ) + ( (lambda/m) * ( Theta1(:,2:y) ) ) ;

Theta2_grad(:,1) = (1/m) * GDelta2(:,1);
Theta2_grad(:,2:c) = ( (1/m) * ( GDelta2(:,2:c) ) ) + ( (lambda/m) * ( Theta2(:,2:c)) ) ;








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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
