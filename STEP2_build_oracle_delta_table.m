%% STEP2_BUILD_ORACLE_DELTA_TABLE
% Purpose:
%   Build the clean oracle calibration table from scratch using the
%   layer based ASBM generator.
%
% For each selected configuration (n, L, d), this script:
%   1. Generates one fixed vector of layer widths n_l.
%   2. Sweeps beta_star over the chosen grid.
%   3. Records the oracle threshold beta_oracle and minimum error err_min.
%   4. Converts beta_oracle into delta_oracle via
%         rho = n^{-beta} (log n)^{1+delta}.
%
% Output table columns:
%   n, L, d, rho, errMin, betaOracle, deltaOracle
%
% Main output folder:
%   organized_simulations/results/step2_oracle_table

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

case_table = sim_case_grid_adaptive();

outdir = fullfile(this_dir, 'results', 'step2_oracle_table');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

nCases = height(case_table);

oracle_rows = cell(nCases, 7);

for i = 1:nCases
    n = case_table.n(i);
    L = case_table.L(i);
    d = case_table.d(i);

    fprintf('\n====================================================\n');
    fprintf('Case %d/%d: n=%d, L=%d, d=%.2f\n', i, nCases, n, L, d);
    fprintf('====================================================\n');

    Res = sim_oracle_beta_curve(n, L, d, opts);

    oracle_rows(i,:) = { ...
        Res.n, Res.L, Res.d, Res.rho, ...
        Res.err_min, Res.beta_oracle, Res.delta_oracle};

    save(fullfile(outdir, sprintf('oracle_curve_n%d_L%d_d%03d.mat', ...
        n, L, round(100*d))), 'Res');

    fprintf('beta_oracle = %.3f | err_min = %.4f | delta_oracle = %.4f\n', ...
        Res.beta_oracle, Res.err_min, Res.delta_oracle);
end

T_oracle = cell2table(oracle_rows, 'VariableNames', ...
    {'n','L','d','rho','errMin','betaOracle','deltaOracle'});

writetable(T_oracle, fullfile(outdir, 'oracle_delta_table.csv'));
save(fullfile(outdir, 'oracle_delta_table.mat'), 'T_oracle', 'opts', 'case_table');

fprintf('\nOracle calibration table saved to: %s\n', outdir);
