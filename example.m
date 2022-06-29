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

% Number of students.
% Must be evenly divisible by 3 schools.
n = 100 * 12;

% Generate GPAs.
gpa = normrnd(2.5,0.5,n,1);
gpa(gpa < 1) = 1;
gpa(gpa > 4) = 4;

% Split students into 3 high schools.
school = zeros(n,3);
school(1:(n/3), 1) = 1; % Jennings
school((n/3+1):(2*n/3), 2) = 1; % Gateway
school((2*n/3+1):n, 3) = 1; % Burroughs

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

%% Fixed Effects Model (GPA only)

model_fixed = struct;

% The observations are SAT score.
Y = sat;
model_fixed.Y = Y;

% Construct fixed effects design matrix.
X = [ones(n,1), gpa];
model_fixed.X = X;

% Fit the fixed effects model.
model_fixed.B = pinv(X)*Y;

% Find the standard errors.
model_fixed.SE = sqrt(diag(pinv(X)*pinv(X)'*sum((Y-X*model_fixed.B).^2)/size(Y,1)));

% Compute T-values.
model_fixed.T = model_fixed.B ./ model_fixed.SE;

% Scatterplot
figure; hold on;
scatter_h = scatter(gpa, sat, 'k');
xlabel('Grade Point Average'); ylabel('SAT Score');
xrange = [0.9, 4.1]; xlim(xrange); ylim([350,1650]);
line(xrange, model_fixed.B(1) + xrange.*model_fixed.B(2), 'Color', scatter_h.CData, 'LineStyle', '--');
title({['Fixed Effects B = [' num2str(model_fixed.B(1), '%0.0f') '; ' num2str(model_fixed.B(2), '%0.0f') ']']; ['SE = [' num2str(model_fixed.SE(1), '%0.0f') '; ' num2str(model_fixed.SE(2), '%0.0f') '], T = [' num2str(model_fixed.T(1), '%0.2f') ', ' num2str(model_fixed.T(2), '%0.2f') ']']}, 'interpreter', 'none');

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
figure; hold on;
scatter_h1 = scatter(gpa(logical(school(:,1))), sat(logical(school(:,1))));
scatter_h2 = scatter(gpa(logical(school(:,2))), sat(logical(school(:,2))));
scatter_h3 = scatter(gpa(logical(school(:,3))), sat(logical(school(:,3))));
xlabel('Grade Point Average'); ylabel('SAT Score');
xrange = [0.9, 4.1]; xlim(xrange); ylim([350,1650]);
line_fixed = line(xrange, model_fixed.B(1) + xrange.*model_fixed.B(2), 'Color', 'black', 'LineStyle', '--');
line_marginal = line(xrange, model_int.B(1) + xrange.*model_int.B(2), 'Color', 'black');
line(xrange, model_int.B(1) + model_int.u(1) + xrange.*model_int.B(2), 'Color', scatter_h1.CData);
line(xrange, model_int.B(1) + model_int.u(2) + xrange.*model_int.B(2), 'Color', scatter_h2.CData);
line(xrange, model_int.B(1) + model_int.u(3) + xrange.*model_int.B(2), 'Color', scatter_h3.CData);
legend([scatter_h3, scatter_h2, scatter_h1, line_marginal, line_fixed], 'Burroughs', 'Gateway', 'Jennings', 'Marginal', 'Fixed Only', 'Location', 'Northwest');
title({['Random Intercept, B = [' num2str(model_int.B(1), '%0.0f') '; ' num2str(model_int.B(2), '%0.0f') ']']; ['SE = [' num2str(model_int.SE(1), '%0.0f') '; ' num2str(model_int.SE(2), '%0.2f') '], T = [' num2str(model_int.T(1), '%0.2f') ', ' num2str(model_int.T(2), '%0.0f') ']']}, 'interpreter', 'none');

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
figure; hold on;
scatter_h1 = scatter(gpa(logical(school(:,1))), sat(logical(school(:,1))));
scatter_h2 = scatter(gpa(logical(school(:,2))), sat(logical(school(:,2))));
scatter_h3 = scatter(gpa(logical(school(:,3))), sat(logical(school(:,3))));
xlabel('Grade Point Average'); ylabel('SAT Score');
xrange = [0.9, 4.1]; xlim(xrange); ylim([350,1650]);
line_fixed = line(xrange, model_fixed.B(1) + xrange.*model_fixed.B(2), 'Color', 'black', 'LineStyle', '--');
line_marginal = line(xrange, model_slope.B(1) + xrange.*model_slope.B(2), 'Color', 'black');
line(xrange, model_slope.B(1) + xrange.*(model_slope.B(2) + model_slope.u(1)), 'Color', scatter_h1.CData);
line(xrange, model_slope.B(1) + xrange.*(model_slope.B(2) + model_slope.u(2)), 'Color', scatter_h2.CData);
line(xrange, model_slope.B(1) + xrange.*(model_slope.B(2) + model_slope.u(3)), 'Color', scatter_h3.CData);
legend([scatter_h3, scatter_h2, scatter_h1, line_marginal, line_fixed], 'Burroughs', 'Gateway', 'Jennings', 'Marginal', 'Fixed Only', 'Location', 'Northwest');
title({['Random Slope, B = [' num2str(model_slope.B(1), '%0.0f') '; ' num2str(model_slope.B(2), '%0.0f') ']']; ['SE = [' num2str(model_slope.SE(1), '%0.0f') '; ' num2str(model_slope.SE(2), '%0.0f') '], T = [' num2str(model_slope.T(1), '%0.2f') ', ' num2str(model_slope.T(2), '%0.2f') ']']}, 'interpreter', 'none');

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
figure; hold on;
scatter_h1 = scatter(gpa(logical(school(:,1))), sat(logical(school(:,1))));
scatter_h2 = scatter(gpa(logical(school(:,2))), sat(logical(school(:,2))));
scatter_h3 = scatter(gpa(logical(school(:,3))), sat(logical(school(:,3))));
xlabel('Grade Point Average'); ylabel('SAT Score');
xrange = [0.9, 4.1]; xlim(xrange); ylim([350,1650]);
line_fixed = line(xrange, model_fixed.B(1) + xrange.*model_fixed.B(2), 'Color', 'black', 'LineStyle', '--');
line_marginal = line(xrange, model_both.B(1) + xrange.*model_both.B(2), 'Color', 'black');
line(xrange, model_both.B(1) + model_both.u(1) + xrange.*(model_both.B(2) + model_both.u(4)), 'Color', scatter_h1.CData);
line(xrange, model_both.B(1) + model_both.u(2) + xrange.*(model_both.B(2) + model_both.u(5)), 'Color', scatter_h2.CData);
line(xrange, model_both.B(1) + model_both.u(3) + xrange.*(model_both.B(2) + model_both.u(6)), 'Color', scatter_h3.CData);
legend([scatter_h3, scatter_h2, scatter_h1, line_marginal, line_fixed], 'Burroughs', 'Gateway', 'Jennings', 'Marginal', 'Fixed Only', 'Location', 'Northwest');
title({['Random Intercept + Slope B = [' num2str(model_both.B(1), '%0.0f') '; ' num2str(model_both.B(2), '%0.0f') ']']; ['SE = [' num2str(model_both.SE(1), '%0.0f') '; ' num2str(model_both.SE(2), '%0.0f') '], T = [' num2str(model_both.T(1), '%0.2f') ', ' num2str(model_both.T(2), '%0.2f') ']']}, 'interpreter', 'none');

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
figure; hold on;
scatter_h1 = scatter(gpa(logical(school(:,1))), sat(logical(school(:,1))));
scatter_h2 = scatter(gpa(logical(school(:,2))), sat(logical(school(:,2))));
scatter_h3 = scatter(gpa(logical(school(:,3))), sat(logical(school(:,3))));
xlabel('Grade Point Average'); ylabel('SAT Score');
xrange = [0.9, 4.1]; xlim(xrange); ylim([350,1650]);
line_fixed = line(xrange, model_fixed.B(1) + xrange.*model_fixed.B(2), 'Color', 'black', 'LineStyle', '--');
line_marginal = line(xrange, model_reml.B(1) + xrange.*model_reml.B(2), 'Color', 'black');
line(xrange, model_reml.B(1) + model_reml.u(1) + xrange.*(model_reml.B(2) + model_reml.u(4)), 'Color', scatter_h1.CData);
line(xrange, model_reml.B(1) + model_reml.u(2) + xrange.*(model_reml.B(2) + model_reml.u(5)), 'Color', scatter_h2.CData);
line(xrange, model_reml.B(1) + model_reml.u(3) + xrange.*(model_reml.B(2) + model_reml.u(6)), 'Color', scatter_h3.CData);
legend([scatter_h3, scatter_h2, scatter_h1, line_marginal, line_fixed], 'Burroughs', 'Gateway', 'Jennings', 'Marginal', 'Fixed Only', 'Location', 'Northwest');
title({['Random Intercept + Slope (REML) B = [' num2str(model_reml.B(1), '%0.0f') '; ' num2str(model_reml.B(2), '%0.0f') ']'], ['SE = [' num2str(model_reml.SE(1), '%0.0f') '; ' num2str(model_reml.SE(2), '%0.0f') '], T = [' num2str(model_reml.T(1), '%0.2f') ', ' num2str(model_reml.T(2), '%0.2f') ']']});
