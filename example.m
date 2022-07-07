% Generate example/test data for a mixed effects model.
%
% Premise: A scientist wants to test if SAT score is correlated with high
% school GPA.  She has collected data from 120 students attending 3 high
% schools in St. Louis: Jennings (a public high school plagues by poverty),
% Gateway Science Academy (a charter school), and Burroughs (an elite
% college prep school). Due to data being collected during the COVID-19
% pandemic, students attended school using Zoom, Google Meet, Microsoft
% Teams, or in-person.
%
% GPA ranges from 1.0 to 4.0 with an average of about 3.0.
% SAT scores range from 400 (lowest possible score) to 1600 (perfect).
% Both are (supposed) to be normally distributed.

%% Generate the Data

% Number of students in each school.
n_jennings = 1000;
n_gateway = 800;
n_burroughs = 500;
n = n_jennings + n_gateway + n_burroughs;

% Generate GPAs.
gpa = normrnd(2.5,0.5,n,1);
gpa(gpa < 1) = 1;
gpa(gpa > 4) = 4;

% Split students into 3 high schools.
school = zeros(n,3);
school(1:n_jennings, 1) = 1; % Jennings
school((n_jennings+1):(n_jennings+n_gateway), 2) = 1; % Gateway
school((n_jennings+n_gateway+1):n, 3) = 1; % Burroughs

% Also create a categorical variable to indicate school membership.
% 1 = Jennings
% 2 = Gateway
% 3 = Burroughs
G = sum(school .* [1,2,3], 2);

% Simulate some overall relationship between SAT score and GPA.
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
gpa(school(:,3) == 1) = gpa(school(:,3) == 1) + 0.5;
gpa(gpa > 4) = 4;

% Add in homoskedastic error.
sat = sat + normrnd(0, 50, n, 1);

% Clip data to range of possible SAT scores.
sat(sat < 400) = 400;
sat(sat > 1600) = 1600;

% Store data in a closure for easy plotting of results.
plot_clusters = make_plot_fn(sat, gpa, school);

%% Fixed Effects Model Without Clusters

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

% Find the standard errors.
model_fixed.SE = sqrt(diag(pinv(X)*pinv(X)'*sum((Y-X*model_fixed.B).^2)/size(Y,1)));

% Compute T-values.
model_fixed.T = model_fixed.B ./ model_fixed.SE;

% Scatterplot
plot_clusters(model_fixed.B);
title('Fixed Effects Only (No Clusters)');

% Compute Standard Error for Fixed Effects Model Using Sandwich Estimator
resid = model_fixed.Y - model_fixed.X * model_fixed.B;
model_fixed.swe_covB = swe(Xpinv, resid, G);
clear resid Xpinv

% Start a table to track and compare estimates of B and its standard error
% using different method.
comparison_tbl = table(repmat(model_fixed.B', 2, 1), [model_fixed.SE'; sqrt(diag(model_fixed.swe_covB.block))'], 'VariableNames', {'B', 'SE'}, 'RowNames', {'Fixed Only', 'Fixed + Block SwE'});

%% Mixed Effects Model, Random Intercept for School

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

%% Mixed Effects Model, Random Slope for School

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

%% Mixed Effects Model, Random Intercept and Centered Slope for School

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

% We can extract the standard error from the diagonal of the
% variance-covariance matrix for B.
model_both.SE = sqrt(diag(model_both.B_cov));

% Un-center by adding the mean back the estimates of the fixed and random
% effect intercepts.  Note this does *not* un-center the estimates of the
% random effect sigmas.
model_both.B(1) = model_both.B(1) - model_both.B(2).*mean(gpa);
model_both.u(1:3) = model_both.u(1:3) - model_both.u(4:6).*mean(gpa);

model_both.T = model_both.B ./ model_both.SE;

% Scatterplot
plot_clusters(model_both.B, model_both.u);
title('Random Interecept + Slope, MoM');

% Add to comparison table.
comparison_tbl = [comparison_tbl; table(model_both.B', model_both.SE', 'VariableNames', {'B', 'SE'}, 'RowNames', {'MoM Intercept + Slope'})];

%% Use REML, Random Intercept + Centered Slope
G = sum(school .* [1,2,3], 2);
reml = fitlmematrix(X_centered,Y,X_centered,G, 'FitMethod', 'REML');
model_reml = struct;
model_reml.X = X_centered;
model_reml.Y = Y;
model_reml.B = fixedEffects(reml);
model_reml.u = randomEffects(reml); model_reml.u = [model_reml.u(1); model_reml.u(3); model_reml.u(5); model_reml.u(2); model_reml.u(4); model_reml.u(6)];
[psi, mse] = covarianceParameters(reml);
model_reml.sigmas = diag(psi{1}); clear psi
model_reml.mse = mse; clear mse
model_reml.SE = sqrt(diag(reml.CoefficientCovariance));
clear X_centered Y

% Un-center by adding the mean back the estimates of the fixed and random
% effect intercepts.  Note this does *not* un-center the estimates of the
% random effect sigmas.
model_reml.B(1) = model_reml.B(1) - model_reml.B(2).*mean(gpa);
model_reml.u(1:3) = model_reml.u(1:3) - model_reml.u(4:6).*mean(gpa);

% Compute t-values.
model_reml.T = model_reml.B ./ model_reml.SE;

% Scatterplot
plot_clusters(model_reml.B, model_reml.u);
title('Random Intercept + Slope (REML)');

% Add to the comparison table.
comparison_tbl = [comparison_tbl; table(model_reml.B', model_reml.SE', 'VariableNames', {'B', 'SE'}, 'RowNames', {'REML Intercept + Slope'})];

%% Fixed Effects with Clusters

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
% Find the standard errors.
SE = sqrt(diag(pinv(X)*pinv(X)'*sum((Y-X*B).^2)/size(Y,1)));
model_marginal_both = struct('X', X, 'Y', sat, 'B', B, 'u', u, 'SE', SE);
clear X Y B u SE T

% Compute Standard Error for Fixed Effects Model Using Sandwich Estimator
Xpinv = pinv(model_marginal_both.X(:, 1:2));
resid = model_marginal_both.Y - model_marginal_both.X(:, 1:2) * model_marginal_both.B(1:2);
model_marginal_both.swe_covB = swe(Xpinv, resid, G);
clear resid Xpinv
model_marginal_both.SE_swe = sqrt(diag(model_marginal_both.swe_covB.block));
model_marginal_both.T = model_marginal_both.B(1:2) ./ model_marginal_both.SE_swe(1:2);

% Add to comparison table.
comparison_tbl = [comparison_tbl; table(repmat(model_marginal_both.B(1:2)', 2, 1), [model_marginal_both.SE(1:2)'; sqrt(diag(model_marginal_both.swe_covB.block))'], 'VariableNames', {'B', 'SE'}, 'RowNames', {'Marginal Only', 'Marginal + Block SwE'})]

% Scatterplot
plot_clusters(model_marginal_both.B, model_marginal_both.u);
title('Fixed Effects With Clusters (Marginal)');

%% Visualize comparison table.

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
xlabels = {'Fixed Only', 'REML', 'MoM', 'Marginal'};
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