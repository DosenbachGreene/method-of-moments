function [sigmas, mse] = HE_regress(Y, Z, U, sigma_ncols, allow_negative_sigmas)
% Perform Haeseman-Elston regression.
%
% If allow_negative_sigmas == true then a warning will be emitted for any
% negative values of sigma and the resultant value will be set to zero
% instead.

% Compute the preorthogonalized variance-covariance matrices for each
% grouping in sigma_ncols inside the null space of the fixed effects design
% matrix X.  Then vectorize the upper triangular portion of each matrix in
% preparation for Haesman-Elston (HE) regression.
XU = zeros((size(U,2)*size(U,2)-size(U,2))/2+size(U,2), length(sigma_ncols) + 1);
i = 1;
for j=1:length(sigma_ncols)
    % Note: if using BLAS, precompute UtZ = U'*Z and call dsyrk() for fast
    % computation of the triangular part of UtZ * UtZ'.
    UtZ = U' * Z(:,i:(i+sigma_ncols(j)-1));
    XU(:,j) = vec_triu(UtZ * UtZ');
    i = i + sigma_ncols(j);
end
clear UtZ;

% Add the homoskedastic mean squared error sigma as the last column in XU.
% Note that U'*U == eye(size(U,2))
% And thus U'*eye(size(Y,1))*U == eye(size(U,2))
XU(:,end) = vec_triu(eye(size(U,2)));

% Transform the observed variance-covariance into the null space and
% vectorize its upper triangular portion.
UtY = U'*Y;
YU = vec_triu(UtY * UtY'); % Again, use dsyrk() if possible.
clear UtY;

% Perform the regression.
sigmas = qr_pinv(XU)*YU;

% Make sure all the variance components are positive.  Negative variance is
% impossible, so if we get a negative variance then something has gone
% wrong, usually due to colinearity in XU.  However, sometimes small
% negative variances will arise when the true value of the sigma is close
% to zero.
if nargin < 5
    % Default to generating an error for negative variances.
    allow_negative_sigmas = false;
end
if allow_negative_sigmas
    negative_sigmas = sigmas < 0;
    if any(negative_sigmas)
        warning(['Setting negative sigma to zero, sigmas = ' mat2str(sigmas)]);
        sigmas(negative_sigmas) = 0;
    end
else
    assert(all(sigmas >= 0));
end

% Return the MSE component separately.
[sigmas, mse] = deal(sigmas(1:end-1), sigmas(end));

end

function v = vec_triu(m)
% Vectorize the upper triangular part of the matrix m.

assert(size(m,1) == size(m,2));
n = size(m,1);
v = zeros((n*n-n)/2+n, 1);
i = 1;
for j=1:n % iterate over columns in m
    v(i:(i+j-1)) = m(1:j, j);
    i = i + j;
end

end