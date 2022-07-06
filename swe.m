function covB = swe(Xpinv, resid, groups)
% Covariance matrix of the fixed effects using the sandwich estimator.
%
% Compute the variance-covariance matrix of some marginal, fixed effects B
% using the identity-weighted Huber-White Sandwich Estimator under
% different assumptions about the variance-covariance structure of the
% ordinary least squares residuals.  Takes the pseudoinverse of the design
% matrix X, Xpinv, and the residuals of fixed effects regression, resid, as
% inputs.  The structure of the residuals is specified by the caterogical
% variable groups.
%
% Briefly, the sanwich estimator is given by:
% B = inv(X'*W*X)*X'*W*Y
% covB = inv(X'*W*X) * X'*W*V*W*X * inv(X'*W*X)
% Where V is the variance-covariance structure of the residuals (encoding
% group or cluster membership) and W is a weighting matrix.  In a mixed
% effects model the weighting matrix W = inv(V).
%
% In this simplest form of the sandwich estimator, we take V equal to some
% empirical estimate of the variance-covariance structure of the residuals
% given by some constraint on V = resid'*resid, see comments within code
% below for explanations of three common constraints.  Furthermore, we find
% that the sanwich estimator still produces asymptotically-correct
% estimates of the covB (but not B itself!) when W is misspecified.
% Therefore, for simplicity, we take W = I (the identity matrix).  Then the
% sandwich estimator simplifies to:
% covB = inv(X'*X) * X'*V*X * inv(X'*X) = pinv(X) * V * pinv(X)'
%
% For further details, refer to the software package:
% https://www.nisox.org/Software/SwE/
% And to Bryan Guillaume thesis work:
% Guillaume B, Hua X, Thompson PM, Waldorp L, Nichols TE; Alzheimer's Disease Neuroimaging Initiative. Fast and accurate modelling of longitudinal and repeated measures neuroimaging data. Neuroimage. 2014 Jul 1;94:287-302. doi: 10.1016/j.neuroimage.2014.03.029
%
% Also see commentary by David Freedman for caveats:
% Freedman DA. On The So-Called "Huber Sandwich Estimator" and "Robust Standard Errors." The American Statistician. 2006;60(4):299-302. doi:10.1198/000313006x152207
% Note that the following common claim regarding the sandwich estimator is
% probably *not* completely accurate:
% "Despite having no explicit specification, all possible random effects
% are accounted for through the use of an unstructured error covariance."

% Prepare the grouping variable.
if ~iscategorical(groups)
    groups = categorical(groups);
end
group_ids = categories(groups);

% The block version of the sandwich estimator requires at least two groups.
% When there is only one group (i.e. the entire variance-covariance matrix
% of the residuals is one big, unconstrained block) the sandwich estimator
% of covB shrinks to zero.
assert(length(group_ids) > 1);

% Create a random effect design matrix of ones and zeros encoding group
% with one column for each group.
Z = zeros(size(Xpinv,2), length(group_ids));
for i=1:length(group_ids)
    Z(groups == group_ids(i), i) = 1;
end

% Compute covariance of fixed effect (marginal) estimates (Betas, B) using
% different assumptions.

%% Assumption 1:
% Block diagonal covariance where errors within each group can be
% correlated, but errors between groups are independent.

% Compose the empirical variance-covariance matrix.
V = (Z*Z') .* (resid*resid');

% Use the sandwich estimator.
covB_block = Xpinv*V*Xpinv';

%% Assumption 2:
% Errors are heteroskedastic but independent, i.e. V is diagonal.

% Compose the diagonal of the variance covariance matrix.
V = diag(V);

% Use the sandwich estimator.
% Equivalent to Xpinv * diag(V) * Xpinv
% Note, if you only need the diagonal (e.g. to get the standard error for a
% t-test) then you could do: sum(Xpinv.*V'.*Xpinv, 2)
covB_diag = Xpinv.*V'*Xpinv';

%% Assumption 3:
% Errors are pooled heteroskedastic.  All errors within the same group have
% the same, pooled variance.

% Compose the diagonal of the variance covariance matrix.
for i=1:length(group_ids)
    V(groups == group_ids(i)) = mean(V(groups == group_ids(i)));
end

% Use the sandwich estimator.
% Equivalent to Xpinv * diag(V) * Xpinv
% Note, if you only need the diagonal (e.g. to get the standard error for a
% t-test) then you could do: sum(Xpinv.*V'.*Xpinv, 2)
covB_pooled_diag = Xpinv.*V'*Xpinv';

% Return results as a structure.
covB = struct( ...
    'diag', covB_diag, ...
    'pooled_diag', covB_pooled_diag, ...
    'block', covB_block ...
);

end