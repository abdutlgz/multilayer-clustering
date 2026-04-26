%% STEP3_FIT_DELTA_RULE_AND_COMPARE
% Purpose:
%   Fit an empirical regression rule for delta from the clean oracle table,
%   convert it to beta_empirical, and compare:
%       baseline error
%       empirical-threshold error
%       oracle-threshold error
%
% Required input:
%   organized_simulations/results/step2_oracle_table/oracle_delta_table.csv
%
% Regression model:
%   deltaOracle ~ b0 + b1*log(n) + b2*log(L) + b3*log(d)
%
% Output table columns:
%   n, L, d, rho, deltaOracle, deltaHat, betaOracle, betaEmpirical,
%   errOracle, errEmpirical, errBaseline
%
% Main output folder:
%   organized_simulations/results/step3_delta_rule

clear; clc; close all;
rng(1);

this_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(this_dir);
addpath(this_dir);
addpath(root_dir);

opts = struct();
opts.M = 3;
opts.K = 3;
opts.c = 0;
opts.w = 1;
opts.Nruns = 50;
opts.beta_min = 0.30;
opts.beta_max = 1.10;
opts.beta_star_list = 0.30:0.05:1.10;

indir = fullfile(this_dir, 'results', 'step2_oracle_table');
if ~exist(fullfile(indir, 'oracle_delta_table.csv'), 'file')
    legacy_indir = fullfile(this_dir, 'organized_simulations', 'results', 'step2_oracle_table');
    if exist(fullfile(legacy_indir, 'oracle_delta_table.csv'), 'file')
        indir = legacy_indir;
    end
end

outdir = fullfile(this_dir, 'results', 'step3_delta_rule');

if ~exist(outdir, 'dir')
    mkdir(outdir);
end

T_oracle = readtable(fullfile(indir, 'oracle_delta_table.csv'));

X = [ones(height(T_oracle),1), log(T_oracle.n), log(T_oracle.L), log(T_oracle.d)];
y = T_oracle.deltaOracle;
b = X \ y;

deltaHat = X * b;
betaEmp = ((1 + deltaHat) .* log(log(T_oracle.n)) - log(T_oracle.rho)) ./ log(T_oracle.n);
betaEmp = max(min(betaEmp, max(opts.beta_star_list)), min(opts.beta_star_list));

nCases = height(T_oracle);
errOracle = zeros(nCases, 1);
errEmpirical = zeros(nCases, 1);
errBaseline = zeros(nCases, 1);

for i = 1:nCases
    n = T_oracle.n(i);
    L = T_oracle.L(i);
    d = T_oracle.d(i);
    betaOracle = T_oracle.betaOracle(i);
    betaEmp_i = betaEmp(i);

    [nL_vec, ~] = gen_nl_beta(n, L, opts.beta_min, opts.beta_max);
    nL_vec = max(nL_vec, opts.K);
    beta_vec = log(nL_vec) ./ log(n);

    idx_active_oracle = find(beta_vec >= betaOracle);
    idx_nonactive_oracle = find(beta_vec < betaOracle);

    idx_active_emp = find(beta_vec >= betaEmp_i);
    idx_nonactive_emp = find(beta_vec < betaEmp_i);

    errs_oracle = zeros(1, opts.Nruns);
    errs_emp = zeros(1, opts.Nruns);
    errs_base = zeros(1, opts.Nruns);

    fprintf('\n====================================================\n');
    fprintf('Case %d/%d: n=%d, L=%d, d=%.2f\n', i, nCases, n, L, d);
    fprintf('betaOracle = %.3f | betaEmpirical = %.3f\n', betaOracle, betaEmp_i);
    fprintf('====================================================\n');

    for r = 1:opts.Nruns
        rng(r);
        [A, ~, ~, ~, ~, label] = ASBM_varNL_layerB( ...
            n, nL_vec, opts.K, L, opts.M, opts.c, d, opts.w);

        errs_oracle(r) = sim_split_pipeline_error( ...
            A, label, idx_active_oracle, idx_nonactive_oracle, opts.M, opts.K, n, L);
        errs_emp(r) = sim_split_pipeline_error( ...
            A, label, idx_active_emp, idx_nonactive_emp, opts.M, opts.K, n, L);
        errs_base(r) = sim_baseline_error(A, label, opts.M, opts.K, L);
    end

    errOracle(i) = mean(errs_oracle);
    errEmpirical(i) = mean(errs_emp);
    errBaseline(i) = mean(errs_base);
end

T_compare = T_oracle(:, {'n','L','d','rho','errMin','betaOracle','deltaOracle'});
T_compare.deltaHat = deltaHat;
T_compare.betaEmpirical = betaEmp;
T_compare.errOracle = errOracle;
T_compare.errEmpirical = errEmpirical;
T_compare.errBaseline = errBaseline;

writetable(T_compare, fullfile(outdir, 'delta_rule_comparison_table.csv'));
save(fullfile(outdir, 'delta_rule_comparison_table.mat'), ...
    'T_compare', 'b', 'opts', 'T_oracle');

coef_table = table(b(1), b(2), b(3), b(4), ...
    'VariableNames', {'b0','b1','b2','b3'});
writetable(coef_table, fullfile(outdir, 'delta_rule_coefficients.csv'));

fig = figure('Color', 'w');
hold on; box on;
scatter(T_compare.deltaOracle, T_compare.deltaHat, 70, 'filled');
lo = min([T_compare.deltaOracle; T_compare.deltaHat]);
hi = max([T_compare.deltaOracle; T_compare.deltaHat]);
plot([lo hi], [lo hi], 'k--', 'LineWidth', 1.5);
grid on;
set(gca, 'FontSize', 15, 'LineWidth', 1.2);
xlabel('Oracle \delta', 'FontSize', 18);
ylabel('Predicted \delta', 'FontSize', 18);
title('Empirical regression for \delta', 'FontSize', 18);
exportgraphics(fig, fullfile(outdir, 'delta_oracle_vs_hat.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outdir, 'delta_oracle_vs_hat.pdf'), 'ContentType', 'vector');
close(fig);

fprintf('\nFitted delta rule:\n');
fprintf('deltaHat = %.6f + %.6f*log(n) + %.6f*log(L) + %.6f*log(d)\n', ...
    b(1), b(2), b(3), b(4));
fprintf('Saved comparison outputs to: %s\n', outdir);
