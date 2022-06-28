function [sqrtm_inv_A, inv_A] = sqrtm_inv_pd(A, eps)
% Compute the inverse and the principal square root of the inverse of the
% positive definite matrix A using Cholesky-based eigendecomposition.
% Faster than sqrtm(inv()) but only works on positive definite matrices.

% Matlab automatically uses the cholesky algorithm for symmetric matrices.
% For other linear algebra libraries use eigh to select cholesky algorithm.
% The 'vector' option gets the eigenvalues D as a vector.
[V, D] = eig(A, 'vector');

%% Check that A is, in fact, positive definite.

% Eigenvalues must be all positive.
assert(all(D > 0));

% Eigenvalues must be real, otherwise matrix was not symmetric.
% However, due to numerical precision issues, the matrix A may be slightly
% asymmetric leading to slightly complex eigenvalues.
if nargin < 2
    eps = 0.0001; % default precision for determining rank
end
assert(all(abs(imag(D)) < eps));
% After verifying the matrix was approximately symmetric we can safely
% discard the imaginary parts of the eigenvectors and eigenvalues.
V = real(V);
D = real(D);

%% Proceed with computation.

% Compute the inverse by inverting the diagonal.
D = 1 ./ D;
inv_A = V .*D' * V';

% Compute the square root of the inverse by taking the square root of the
% diagonal.
sqrtm_inv_A = V .* sqrt(D') * V';

end