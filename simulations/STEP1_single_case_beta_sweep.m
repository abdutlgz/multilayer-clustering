%% STEP1_SINGLE_CASE_BETA_SWEEP
% Purpose:
%   Start from one chosen configuration (n, L, d), sweep the active-layer
%   threshold beta_star, and verify that the total error curve is U-shaped
%   with a nontrivial minimum.
%
% What this script does:
%   1. Generates one fixed vector of layer widths n_l.
%   2. Sweeps beta_star over a user-specified grid.
%   3. For each beta_star, repeatedly simulates corrected ASBM data.
%   4. Runs the split-and-classify pipeline and averages the total error.
%   5. Saves the oracle threshold and the full error curve.
%
% Main output folder:
%   organized_simulations/results/step1_single_case

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

% Choose one representative configuration.
n = 300;
L = 50;
d = 0.25;

outdir = fullfile(this_dir, 'results', 'step1_single_case');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

Res = sim_oracle_beta_curve(n, L, d, opts);

T_curve = table(Res.beta_star_list(:), Res.err_curve(:), Res.active_count(:), ...
    'VariableNames', {'beta_star','err_total','n_active'});
writetable(T_curve, fullfile(outdir, 'single_case_beta_curve.csv'));

save(fullfile(outdir, 'single_case_beta_curve.mat'), 'Res', 'opts');

fig = figure('Color', 'w');
hold on; box on;
plot(Res.beta_star_list, Res.err_curve, '-o', 'LineWidth', 2.5, 'MarkerSize', 7);
plot(Res.beta_oracle, Res.err_min, 'kx', 'LineWidth', 3, 'MarkerSize', 13);
grid on;
set(gca, 'FontSize', 15, 'LineWidth', 1.2);
xlabel('\beta threshold', 'FontSize', 18);
ylabel('Mean total error', 'FontSize', 18);
title(sprintf('Oracle threshold sweep: n=%d, L=%d, d=%.2f', n, L, d), 'FontSize', 18);
legend({'Error curve', 'Oracle minimum'}, 'Location', 'best', 'FontSize', 12);

exportgraphics(fig, fullfile(outdir, 'single_case_beta_curve.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outdir, 'single_case_beta_curve.pdf'), 'ContentType', 'vector');
close(fig);

fprintf('\nSingle-case sweep finished.\n');
fprintf('n=%d, L=%d, d=%.2f, rho=%.3f\n', Res.n, Res.L, Res.d, Res.rho);
fprintf('beta_oracle = %.3f\n', Res.beta_oracle);
fprintf('err_min = %.4f\n', Res.err_min);
fprintf('delta_oracle = %.4f\n', Res.delta_oracle);
fprintf('Saved results to: %s\n', outdir);
