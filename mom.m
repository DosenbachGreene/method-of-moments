function model = mom(X, Y, Z, sigma_ncols, allow_negative_sigmas)
% Estimate a mixed effects model using method of moments.
%
% Based on method described in:
% Lin,Z Z., Seal, S., & Basu, S. (2020). Estimating SNP heritability in presence of population substructure in large biobank-scale data. bioRxiv. https://doi.org/10.1101/2020.08.05.236901
% Ge T, Reuter M, Winkler AM, Holmes AJ, Lee PH, Tirrell LS, Roffman JL, Buckner RL, Smoller JW, Sabuncu MR. Multidimensional heritability analysis of neuroanatomical shape. Nat Commun. 2016 Nov 15;7:13291. doi: 10.1038/ncomms13291. PMID: 27845344; PMCID: PMC5116071.

% Default to disallowing negative sigmas.
if nargin < 5
    allow_negative_sigmas = false;
end

% Sanity checks.
% TODO check more preconditions so we can give better error messages.
assert(sum(sigma_ncols) == size(Z,2), "Number of columns for random effect variances in sigma_ncols does not match number of columns in Z.");

% Find the (transposed) null space, U, of X.
U = qr_null(X);

% Perform Haeseman-Elston regression in the null space of X to find the
% method-of-moments based estimates of the sigmas by regressing the
% variance-covariance of Z onto Y.
% We also get the leftover, homoskedastic variance (the mean squared error
% of the marginal model).
[sigmas, mse] = HE_regress(Y, Z, U, sigma_ncols, allow_negative_sigmas);

% Compute the overall variance covariance matrix. We will use the block-
% diagonal grouping matrix G, which we will use later on to compute the
% random effect coefficients u.
G = zeros(sum(sigma_ncols));
i = 1;
for j = 1:length(sigma_ncols)
    sigma_cols = i:(i+sigma_ncols(j)-1);
    G(sigma_cols,sigma_cols) = sigmas(j)*eye(sigma_ncols(j));
    i = i + sigma_ncols(j);
end
V = Z * G * Z' + eye(size(Y,1)) .* mse;

% Compute the inverse of V and the square root of the inverse of V.
% Both will be used for later computations.
[Vinv_sqrt, Vinv] = sqrtm_inv_pd(V);

% Compute the weighted estimate of the fixed effects (betas).
% We want to minimize the L2 norm of:
% sqrt(W)*(Y-X*B)
% Where the weighting matrix W is inv(V) = Vinv.
% We would clasically compute this as:
% B = inv(X'*W*X)*X'*W*Y
% But it will be more numerically stable to use the pseudoinverse:
% B = pinv(sqrt(W)*X)*sqrt(W)*Y
Xpinv = qr_pinv(Vinv_sqrt*X);
B = Xpinv*Vinv_sqrt*Y;
clear Vinv_sqrt

% Compute the variance-covariance structure of B.
% Clasically: inv(X'*Vinv*X)'
B_cov = Xpinv * Xpinv';
clear Xpinv

% Compute the group-specific random effect coefficients, u.
% There is one value for u for each column in Z.
u = G*Z'*Vinv*(Y-X*B);
clear Vinv G

% Return the results as a structure.
model = struct( ...
    'X', X, ...
    'Y', Y, ...
    'Z', Z, ...
    'sigma_ncols', sigma_ncols, ...
    'sigmas', sigmas, ...           % Random effect variance estimates
    'mse', mse, ...                 % Variance estimate for remaining, homoskedastic error (mean squared error)
    'V', V, ...                     % Variance-covariance matrix for Y
    'B', B, ...                     % Fixed effects estimates (betas)
    'B_cov', B_cov, ...             % Variance-covariance matrix of the betas
    'u', u ...                      % Group-specific random effect estimates, one for each column in Z
);

end 