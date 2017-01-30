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

z=X*theta;
b=length(X);
h = zeros(size(z));
P=exp(-z)+1;
h=power(P,-1);
A=(lambda/(2*m))*theta(2:length(theta))'*theta(2:length(theta));
J=(1/m)*(-(y'*log(h)+(1-y)'*log(1-h)))+A;


grad0=(1/m)*(h-y)'*X(:,1);
Q=(lambda/m)*theta(2:length(theta));
P=X;
P(:,1)=[];
grad1=(1/m)*(((h-y)'*P)')+Q;
grad=[grad0;grad1];

% =============================================================

end
