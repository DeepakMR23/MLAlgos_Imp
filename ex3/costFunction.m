function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

yhats = X*theta;
predictions = sigmoid(yhats);
logpred = log(predictions);
compPreds = 1.-predictions;
logcompPreds = log(compPreds);

indCosts = -(y.*logpred)-((1.-y).*logcompPreds);
J = (1/m)*sum(indCosts);

diff = predictions.-y;
diffaug = repmat(diff, 1, size(X,2));

gradind = diffaug.*X;
grad = ((1/m).*sum(gradind))';




% ====================== ======================
% Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
