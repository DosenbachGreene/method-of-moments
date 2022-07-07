% Generate example/test data for a mixed effects model.
%
% Premise: A scientist wants to test if SAT score is correlated with high
% school GPA.  She has collected data from 2300 students attending 3 high
% schools in St. Louis: Jennings (a public high school plagued by poverty),
% Gateway Science Academy (a charter school), and Burroughs (an elite
% college prep school).
%
% GPA ranges from 1.0 to 4.0 with an average of about 2.5.
% SAT scores range from 400 (lowest possible score) to 1600 (perfect).
% Both are (supposed) to be normally distributed.

%% Generate the Data

% Number of students in each school, or "cluster."
% Try playing with having an equal (balanced) vs unequal (unbalanced)
% number of students in each school.
n_jennings = 1000;
n_gateway = 800;
n_burroughs = 500;
n = n_jennings + n_gateway + n_burroughs;

% Generate GPAs.  Trim GPAs outside the range of possible values.
gpa = normrnd(2.5,0.5,n,1);
gpa(gpa < 1) = 1;
gpa(gpa > 4) = 4;

% Encode the three high schools in an n x 3 matrix.  Each high school
% occupies one column.  Values of 1 indicate a student attends the high
% school for the corresponding column.  The matrix is zero everywhere else.
school = zeros(n,3);
school(1:n_jennings, 1) = 1; % Jennings
school((n_jennings+1):(n_jennings+n_gateway), 2) = 1; % Gateway
school((n_jennings+n_gateway+1):n, 3) = 1; % Burroughs

% Also create a categorical variable to indicate school membership.
% 1 = Jennings
% 2 = Gateway
% 3 = Burroughs
G = sum(school .* [1,2,3], 2);

% Simulate some overall ground-truth relationship between SAT score and
% GPA.
intercept = 300;
slope = 200;
sat = intercept + gpa .* slope;

% Mix in the effect of school.
% Comment/uncomment to simulate random intercept, random slope, or random
% intercept + slope models.
school_intercept = [-75; 75; 300];
%school_intercept = [0; 0; 0];
school_slope = [-50; -30; 80];
%school_slope = [0; 0; 0];
sat = sat + school * school_intercept + school.*gpa * school_slope;

% Adjust the "true" slope and intercept to reflect the average cluster
% contributions.
intercept = intercept + mean(school_intercept);
slope = slope + mean(school_slope);

% The GPA of students at Burroughs is inflated.
% This makes the fixed effects only estimate biased.
% Try commenting this out and see what happens to the fixed effects only
% estimate.
gpa(school(:,3) == 1) = gpa(school(:,3) == 1) + 0.5;
gpa(gpa > 4) = 4;

% Add in homoskedastic error.
sat = sat + normrnd(0, 50, n, 1);

% Clip data to range of possible SAT scores.
sat(sat < 400) = 400;
sat(sat > 1600) = 1600;

% Store data in a closure for easy plotting of results.
plot_clusters = make_plot_fn(sat, gpa, school);

%% Fixed Effects Only Model (Without Clusters)

model_fixed = struct;

% The observations are SAT score.
Y = sat;
model_fixed.Y = Y;

% Construct fixed effects design matrix.
X = [ones(n,1), gpa];
model_fixed.X = X;

% Fit the fixed effects model.
Xpinv = pinv(X);
model_fixed.B = pinv(X)*Y;

% Scatterplot
plot_clusters(model_fixed.B);
title('Fixed Effects Only (No Clusters)');

% Find the standard errors.
model_fixed.SE = sqrt(diag(Xpinv*Xpinv'*sum((Y-X*model_fixed.B).^2)/size(Y,1)));

% Compute T-values.
model_fixed.T = model_fixed.B ./ model_fixed.SE;

% Compute Standard Error for Fixed Effects Model Using Sandwich Estimator
resid = model_fixed.Y - model_fixed.X * model_fixed.B;
model_fixed.swe_covB = swe(Xpinv, resid, G);
clear resid Xpinv

% Start a table to track and compare estimates of B and its standard error
% using different method.
comparison_tbl = table(repmat(model_fixed.B', 2, 1), [model_fixed.SE'; sqrt(diag(model_fixed.swe_covB.block))'], 'VariableNames', {'B', 'SE'}, 'RowNames', {'Fixed Only', 'Fixed + Block SwE'});

%% Mixed Effects Model using Method of Moments
% With Random Intercept and Centered Slope for School

% MoM with intercept + slope requires us to center the random effects
% design matrix, Z, to avoid colinearity when performing Haesman-Elston
% regression.  For simplicity we will center the whole model.  If you fail
% to do this you will probably get warnings about negative random effect
% variance parameters (sigmas).

% Center the slope by subtracting away its mean.
X_centered = [ones(n,1), gpa - mean(gpa)];
Z_centered = [school, school .* (gpa - mean(gpa))];

% There are two one random effect variances,
% one for the interecept,
% and one for the slope.
sigma_ncols = [3, 3];

% Fit the mixed effects model using the method of moments.
% Rectify negative random effect variances to zero by passing in
% allow_negative_sigmas = true.
model_both = mom(X_centered, Y, Z_centered, sigma_ncols, true);
clear Z_centered sigma_ncols

% Un-center by adding the mean back the estimates of the fixed and random
% effect intercepts.  Note this does *not* un-center the estimates of the
% random effect sigmas.
model_both.B(1) = model_both.B(1) - model_both.B(2).*mean(gpa);
model_both.u(1:3) = model_both.u(1:3) - model_both.u(4:6).*mean(gpa);

% Scatterplot
plot_clusters(model_both.B, model_both.u);
title('Mixed, Random Interecept + Slope (MoM)');

% We can extract the standard error from the diagonal of the
% variance-covariance matrix for B and compute the T values.
model_both.SE = sqrt(diag(model_both.B_cov));
model_both.T = model_both.B ./ model_both.SE;

% Add to comparison table.
comparison_tbl = [comparison_tbl; table(model_both.B', model_both.SE', 'VariableNames', {'B', 'SE'}, 'RowNames', {'MoM Intercept + Slope'})];

%% Mixed Effects Model Using REML
% With Random Intercept + Centered Slope for School

% Note that we do not need to center the model when using REML, but we will
% do so anyway for ease of comparison with the MoM parameters.

% The syntax for fotlmematrix is a little different.  Instead of passing in
% the actual random effects matrix Z, we pass in a pseudo-Z with one column
% for each random effect.  For a random intercept+slope example, this Z
% turns out to be the same as X!  We then specify the groupings in a
% separate categorical variable, G.
G = sum(school .* [1,2,3], 2);
reml = fitlmematrix(X_centered,Y,X_centered,G, 'FitMethod', 'REML');

% Extract the results.  Note that randomEffects() gives us the random
% effects in a different order than above.  Intead of getting three random
% effects for intercept and three for slope, we get three pairs of
% [intercept, slope].
model_reml = struct;
model_reml.X = X_centered;
model_reml.Y = Y;
model_reml.B = fixedEffects(reml);
model_reml.u = randomEffects(reml); model_reml.u = [model_reml.u(1); model_reml.u(3); model_reml.u(5); model_reml.u(2); model_reml.u(4); model_reml.u(6)];
[psi, mse] = covarianceParameters(reml);
model_reml.sigmas = diag(psi{1}); clear psi
model_reml.mse = mse; clear mse
model_reml.SE = sqrt(diag(reml.CoefficientCovariance));
model_reml.T = model_reml.B ./ model_reml.SE;
clear X_centered Y

% Un-center by adding the mean back the estimates of the fixed and random
% effect intercepts.  Note this does *not* un-center the estimates of the
% random effect sigmas.
model_reml.B(1) = model_reml.B(1) - model_reml.B(2).*mean(gpa);
model_reml.u(1:3) = model_reml.u(1:3) - model_reml.u(4:6).*mean(gpa);

% Scatterplot
plot_clusters(model_reml.B, model_reml.u);
title('Mixed, Random Intercept + Slope (REML)');

% Add to the comparison table.
comparison_tbl = [comparison_tbl; table(model_reml.B', model_reml.SE', 'VariableNames', {'B', 'SE'}, 'RowNames', {'REML Intercept + Slope'})];

%% Fixed Effects with Clusters

% We (obviously) cannot model a *random* effect of intercept and slope in a
% fixed effect model.  Instead we will encode these as fixed effects.  Note
% that in prior examples we specified two fixed effects columns and six
% random effect columns (three for intercept, three for slope).  We cannot
% do the same here because we would end up with a rank deficient design
% matrix.  Instead, we model 6 fixed effect columns total.  The way we
% encode cluster (school) membership is carefully chosen such that the
% first two fixed effect columns will end up being the overall (marginal)
% intercept and slope.
%
% In columns 3-4 of X, encoding the "random" intercept, there are three
% schools to encode in, effectively, 2 columns.  The possible values of
% each column must sum to zero.  For example, look at column 3.  Jennings
% is encoded as 1.  Gateway and Burroughs is each encoded as -0.5. 
% 1 - 0.5 - 0.5 = 0.  (It doesn't matter if the number of students in each
% school balanced or not.)
%
% Columns 5-6 for "random" slope are simply the interaction of column 2
% (marginal slope) with columns 3-4 ("random" interecept).

X = [ones(n,1), gpa, school(:,1).*1.5-0.5, school(:,2).*1.5-0.5, gpa.*(school(:,1).*1.5-0.5), gpa.*(school(:,2).*1.5-0.5)];
Y = sat;
B = pinv(X)*Y;
u = zeros(6,1);
u(1) = B(3)-0.5*B(4);
u(2) = B(4)-0.5*B(3);
u(3) = -0.5*B(3)-0.5*B(4);
u(4) = B(5)-0.5*B(6);
u(5) = B(6)-0.5*B(5);
u(6) = -0.5*B(5)-0.5*B(6);
model_marginal_both = struct('X', X, 'Y', sat, 'B', B, 'u', u);

% Scatterplot
plot_clusters(model_marginal_both.B, model_marginal_both.u);
title('Fixed Effects With Fixed Clusters');

% Standard error without sandwich estimator.
model_marginal_both.SE = sqrt(diag(pinv(X)*pinv(X)'*sum((Y-X*B).^2)/size(Y,1)));
clear X Y B u SE

% Standard Error and T-values Using Sandwich Estimator
Xpinv = pinv(model_marginal_both.X(:, 1:2));
resid = model_marginal_both.Y - model_marginal_both.X(:, 1:2) * model_marginal_both.B(1:2);
model_marginal_both.swe_covB = swe(Xpinv, resid, G);
clear resid Xpinv
model_marginal_both.SE_swe = sqrt(diag(model_marginal_both.swe_covB.block));
model_marginal_both.T = model_marginal_both.B(1:2) ./ model_marginal_both.SE_swe(1:2);

% Add to comparison table.
comparison_tbl = [comparison_tbl; table(repmat(model_marginal_both.B(1:2)', 2, 1), [model_marginal_both.SE(1:2)'; sqrt(diag(model_marginal_both.swe_covB.block))'], 'VariableNames', {'B', 'SE'}, 'RowNames', {'Marginal Only', 'Marginal + Block SwE'})];

%% Visualize comparison table.

% Observe that the fixed effects only (without clusters) estimate of the
% intercept and slope is probably biased (assuming grade inflation is not
% commented out at the top of this file).  This will happen anytime there
% is a correlation between fixed effects and random effects design matrices
% (in this case, gpa is correlated with which school a student attends.)
%
% Observe that, without using the sandwich estimator, the fixed effects
% estimates of the standard error are much too narrow, which would lead to
% inflated T-values and false positive results!  On the other hand, the SwE
% SE tends to be bigger than the REML SE, leading to low efficiency and
% false negative results.  The MoM SE tends to fall somewhere in between.

% Fixed effects without clusters.
B = comparison_tbl('Fixed Only', :).B(1); % estimate of interecept
SE = comparison_tbl('Fixed Only', :).SE(1); % standard error
SE_swe = comparison_tbl('Fixed + Block SwE', :).SE(1); % SwE standard error
boxdata_int = [B-SE_swe, B-SE, B, B+SE, B+SE_swe]; % make a boxplot
B = comparison_tbl('Fixed Only', :).B(2); % estimate of slope
SE = comparison_tbl('Fixed Only', :).SE(2); % standard error
SE_swe = comparison_tbl('Fixed + Block SwE', :).SE(2); % SwE standard error
boxdata_slope = [B-SE_swe, B-SE, B, B+SE, B+SE_swe]; % make a boxplot
clear B SE SE_swe

% REML
B = comparison_tbl('REML Intercept + Slope', :).B(1); % estimate of interecept
SE = comparison_tbl('REML Intercept + Slope', :).SE(1); % standard error
boxdata_int = [boxdata_int; [B-SE, B-SE, B, B+SE, B+SE]]; % make a boxplot
B = comparison_tbl('REML Intercept + Slope', :).B(2); % estimate of slope
SE = comparison_tbl('REML Intercept + Slope', :).SE(2); % standard error
boxdata_slope = [boxdata_slope; [B-SE, B-SE, B, B+SE, B+SE]]; % make a boxplot
clear B SE

% MoM
B = comparison_tbl('MoM Intercept + Slope', :).B(1); % estimate of interecept
SE = comparison_tbl('MoM Intercept + Slope', :).SE(1); % standard error
boxdata_int = [boxdata_int; [B-SE, B-SE, B, B+SE, B+SE]]; % make a boxplot
B = comparison_tbl('MoM Intercept + Slope', :).B(2); % estimate of slope
SE = comparison_tbl('MoM Intercept + Slope', :).SE(2); % standard error
boxdata_slope = [boxdata_slope; [B-SE, B-SE, B, B+SE, B+SE]]; % make a boxplot
clear B SE

% Marginal (fixed effects with clusters).
B = comparison_tbl('Marginal Only', :).B(1); % estimate of interecept
SE = comparison_tbl('Marginal Only', :).SE(1); % standard error
SE_swe = comparison_tbl('Marginal + Block SwE', :).SE(1); % SwE standard error
boxdata_int = [boxdata_int; [B-SE_swe, B-SE, B, B+SE, B+SE_swe]]; % make a boxplot
B = comparison_tbl('Marginal Only', :).B(2); % estimate of slope
SE = comparison_tbl('Marginal Only', :).SE(2); % standard error
SE_swe = comparison_tbl('Marginal + Block SwE', :).SE(2); % SwE standard error
boxdata_slope = [boxdata_slope; [B-SE_swe, B-SE, B, B+SE, B+SE_swe]]; % make a boxplot
clear B SE SE_swe

% Prepare boxplot data for plotting.
boxdata_int = boxdata_int(:, [1 2 2 3 4 4 5])';
boxdata_slope = boxdata_slope(:, [1 2 2 3 4 4 5])';

% Make the plot.
f = figure; f.Position(3) = f.Position(3) * 2;
subplot(1,2,1);
boxplot(boxdata_int, 'Whisker', Inf);
xlabels = {'Fixed Only', 'Mixed REML', 'Mixed MoM', 'Fixed Clusters'};
xticklabels(xlabels);
ylabel('SAT Score');
title('Interecept');
clear boxdata_int
subplot(1,2,2);
h_box = boxplot(boxdata_slope, 'Whisker', Inf);
h_box = handle(h_box);
xticklabels(xlabels);
ylabel('Change in SAT Score per GPA Point');
title('Slope');
h_swe = h_box(2);
h_se = h_box(5);
legend([h_se, h_swe], 'Standard Error', 'SwE Error', 'Location', 'Northeast');
clear h_box h_se h_swe  boxdata_slope xlabels f

%% Additional Examples

%% Mixed Effects Model, Random Intercept for School
%{
% The random effect design matrix is just three columns of ones and zeros
% encoding which school the student goes to.
Z = school;

% There is only one random effect variance, that of the intercept.
sigma_ncols = [3];

% Fit the mixed effects model using the method of moments!
model_int = mom(X, Y, Z, sigma_ncols);
clear Z sigma_ncols;

% We can extract the standard error from the diagonal of the
% variance-covariance matrix for B.
model_int.SE = sqrt(diag(model_int.B_cov));
model_int.T = model_int.B ./ model_int.SE;

% Scatterplot
plot_clusters(model_int.B, [model_int.u; 0; 0; 0]);
title('Random Interecept, MoM');

% Add to comparison table.
comparison_tbl = [comparison_tbl; table(model_int.B', model_int.SE', 'VariableNames', {'B', 'SE'}, 'RowNames', {'MoM Intercept'})];
%}

%% Mixed Effects Model, Random Slope for School
%{

% The random effect design matrix is three columns containing the GPA for
% schools in the column, otherwise zero.
Z = school.*gpa;

% There is only one random effect variance, that of the slope.
sigma_ncols = [3];

% Fit the mixed effects model using the method of moments!
model_slope = mom(X, Y, Z, sigma_ncols);
clear X Z sigma_ncols;

% We can extract the standard error from the diagonal of the
% variance-covariance matrix for B.
model_slope.SE = sqrt(diag(model_slope.B_cov));
model_slope.T = model_slope.B ./ model_slope.SE;

% Scatterplot
plot_clusters(model_slope.B, [0; 0; 0; model_slope.u]);
title('Random Slope, MoM');

% Add to comparison table.
comparison_tbl = [comparison_tbl; table(model_slope.B', model_slope.SE', 'VariableNames', {'B', 'SE'}, 'RowNames', {'MoM Slope'})];
%}