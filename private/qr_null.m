function Xnull = qr_null(X, eps)
% Efficiently compute the transposed null space of X using QR
% decomposition.
%
% X must be full column rank to precision eps (default eps = 0.0001).
% The null space is computed such that Xnull has dimensions:
% size(X,1), size(X,1)-size(X,2)
% And satisfies the following conditions:
% Xnull'*X == 0
% Xnull'*Xnull == eye(size(Xnull,2))
% Xnull*Xnull'== eye(size(Xnull,1)) - X*pinv(X)

% Compute the QR decomposition of X.
[Q, R] = qr(X);

% Check that X is full rank.
if nargin < 2
    eps = 0.0001; % default precision for determining rank
end
if size(X,2) > size(X,1) || any(abs(diag(R)) < eps)
    warning('X is not full rank.');
end

% Obtain the null space of X.
Xnull = Q(:,size(X,2)+1:end);

end