%% STEP4_MAKE_PAPER_FIGURES
% Create the final paper-ready figures and tables from the saved results of
% the layer-specific-B ASBM experiments.
%
% Important conventions used here:
%   beta_star_vec     = layer-width exponents beta_l^*
%   grid_g            = candidate threshold grid
%   beta_hat_oracle   = oracle grid threshold selected using true labels
%   beta_hat_emp      = regression-based threshold (raw value)
%   rho_n             = sparsity parameter
%   calR_tot          = total risk
%   calR_act          = active-layer clustering risk
%   calR_non          = non-active-layer classification risk
%
% Final outputs:
%   Graphs/Simulations/FINAL_ERROR_vs_G_rho_panels.png
%   Graphs/Simulations/FINAL_ERROR_vs_G_size_panels.png
%   Graphs/Simulations/FINAL_METHOD_COMPARISON_bars.png
%   Graphs/Simulations/FINAL_ERROR_DECOMPOSITION.png
%   Graphs/Simulations/FINAL_RELIMP_vs_rho.png
%   Tables/table_beta_calibration_layerB.csv
%   Tables/table_beta_calibration_layerB.tex
%   Tables/table_method_comparison_layerB.csv
%   Tables/table_method_comparison_layerB.tex

clear; clc; close all;

this_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(this_dir);
addpath(this_dir);
addpath(root_dir);

step2_dir = fullfile(this_dir, 'results', 'step2_oracle_table');
step3_dir = fullfile(this_dir, 'results', 'step3_delta_rule');

legacy_step2_dir = fullfile(this_dir, 'organized_simulations', 'results', 'step2_oracle_table');
legacy_step3_dir = fullfile(this_dir, 'organized_simulations', 'results', 'step3_delta_rule');

if ~exist(fullfile(step2_dir, 'oracle_delta_table.csv'), 'file') && ...
        exist(fullfile(legacy_step2_dir, 'oracle_delta_table.csv'), 'file')
    step2_dir = legacy_step2_dir;
end

if ~exist(fullfile(step3_dir, 'delta_rule_comparison_table.csv'), 'file') && ...
        exist(fullfile(legacy_step3_dir, 'delta_rule_comparison_table.csv'), 'file')
    step3_dir = legacy_step3_dir;
end

step2_csv = fullfile(step2_dir, 'oracle_delta_table.csv');
step3_csv = fullfile(step3_dir, 'delta_rule_comparison_table.csv');

if ~exist(step2_csv, 'file')
    error('STEP4:MissingStep2', ...
        'Missing file: %s. Run STEP2_build_oracle_delta_table.m first.', step2_csv);
end

if ~exist(step3_csv, 'file')
    error('STEP4:MissingStep3', ...
        'Missing file: %s. Run STEP3_fit_delta_rule_and_compare.m first.', step3_csv);
end

graph_dir = fullfile(root_dir, 'Graphs', 'Simulations');
table_dir = fullfile(root_dir, 'Tables');
if ~exist(graph_dir, 'dir')
    mkdir(graph_dir);
end
if ~exist(table_dir, 'dir')
    mkdir(table_dir);
end

T_oracle = readtable(step2_csv);
T_compare = readtable(step3_csv);

opts = struct();
opts.M = 3;
opts.K = 3;
opts.c = 0;
opts.w = 1;
opts.Nruns = 50;

%% =========================================================
% Build a consolidated per-case structure
% ==========================================================
Cases = build_case_struct(T_oracle, T_compare, step2_dir);

confirm_checks(Cases);

%% =========================================================
% FIGURE 1: Threshold-risk curves for varying rho_n
% ==========================================================
fig1_targets = [0.10, 0.125, 0.15];
fig1_ids = choose_distinct_cases_by_rho(Cases, 300, 50, fig1_targets);

if any(isnan(fig1_ids))
    fig1_targets = [0.10, 0.125, 0.20];
    fig1_ids = choose_distinct_cases_by_rho(Cases, 300, 50, fig1_targets);
end

make_threshold_panels( ...
    Cases(fig1_ids), ...
    fullfile(graph_dir, 'FINAL_ERROR_vs_G_rho_panels.png'), ...
    'Candidate threshold g', ...
    'Total risk', ...
    @(S) sprintf('n=%d, L=%d, rho_n=%.3f', S.n, S.L, S.rho_n), ...
    true);

%% =========================================================
% FIGURE 2: Threshold-risk curves for increasing problem size
% ==========================================================
fig2_pref = [
    200   50   0.125
    300   75   0.125
    400  100   0.125
];

fig2_ids = find_exact_cases(Cases, fig2_pref);
if any(isnan(fig2_ids))
    fig2_fallback = [
        350   75   0.050
        400  125   0.050
        450  150   0.050
    ];
    fig2_ids = find_exact_cases(Cases, fig2_fallback);
end

make_threshold_panels( ...
    Cases(fig2_ids), ...
    fullfile(graph_dir, 'FINAL_ERROR_vs_G_size_panels.png'), ...
    'Candidate threshold g', ...
    'Total risk', ...
    @(S) sprintf('(n,L)=(%d,%d), rho_n=%.3f', S.n, S.L, S.rho_n), ...
    true);

%% =========================================================
% FIGURE 3: Main method comparison across selected regimes
% ==========================================================
fig3_pref = [
    200   50   0.150
    300   50   0.125
    400  100   0.100
    450  150   0.050
];

fig3_ids = find_closest_requested_cases(Cases, fig3_pref);
fig3_cases = Cases(fig3_ids);

labels = cell(numel(fig3_cases), 1);
bar_vals = zeros(numel(fig3_cases), 3);
for i = 1:numel(fig3_cases)
    labels{i} = sprintf('(%d,%d,%.3f)', fig3_cases(i).n, fig3_cases(i).L, fig3_cases(i).rho_n);
    bar_vals(i,1) = fig3_cases(i).calR_baseline;
    bar_vals(i,2) = fig3_cases(i).calR_emp;
    bar_vals(i,3) = fig3_cases(i).calR_oracle;
end

fig = figure('Color', 'w', 'Position', [100 100 1100 500]);
b = bar(bar_vals, 'grouped');
b(1).FaceColor = [0.35 0.35 0.35];
b(2).FaceColor = [0.88 0.45 0.12];
b(3).FaceColor = [0.15 0.55 0.22];
box on; grid on;
disable_axes_toolbar(fig);
set(gca, 'FontSize', 13, 'LineWidth', 1.1);
set(gca, 'XTick', 1:numel(labels), 'XTickLabel', labels);
xlabel('Regime', 'FontSize', 16);
ylabel('Total risk', 'FontSize', 16);
legend({'Baseline','Empirical','Oracle'}, 'Location', 'best', 'FontSize', 12);
add_bar_labels(b, bar_vals);
exportgraphics(fig, fullfile(graph_dir, 'FINAL_METHOD_COMPARISON_bars.png'), 'Resolution', 300);
close(fig);

%% =========================================================
% FIGURE 4: Error decomposition for empirical and oracle
% This figure recomputes calR_act and calR_non only for the representative
% regimes because those quantities were not stored in STEP2/STEP3 outputs.
% ==========================================================
decomp_emp = zeros(numel(fig3_cases), 2);
decomp_oracle = zeros(numel(fig3_cases), 2);

for i = 1:numel(fig3_cases)
    fprintf('Recomputing decomposition for n=%d, L=%d, rho_n=%.3f\n', ...
        fig3_cases(i).n, fig3_cases(i).L, fig3_cases(i).rho_n);
    [decomp_emp(i,:), decomp_oracle(i,:)] = recompute_decomposition(fig3_cases(i), opts);
end

fig = figure('Color', 'w', 'Position', [100 100 1200 450]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
disable_axes_toolbar(fig);

nexttile;
vals = [decomp_emp(:,1), decomp_oracle(:,1)];
b = bar(vals, 'grouped');
b(1).FaceColor = [0.88 0.45 0.12];
b(2).FaceColor = [0.15 0.55 0.22];
box on; grid on;
set(gca, 'FontSize', 13, 'LineWidth', 1.1);
set(gca, 'XTick', 1:numel(labels), 'XTickLabel', labels);
xlabel('Regime', 'FontSize', 15);
ylabel('Active-layer clustering risk', 'FontSize', 15);
legend({'Empirical','Oracle'}, 'Location', 'best', 'FontSize', 11);

nexttile;
vals = [decomp_emp(:,2), decomp_oracle(:,2)];
b = bar(vals, 'grouped');
b(1).FaceColor = [0.88 0.45 0.12];
b(2).FaceColor = [0.15 0.55 0.22];
box on; grid on;
set(gca, 'FontSize', 13, 'LineWidth', 1.1);
set(gca, 'XTick', 1:numel(labels), 'XTickLabel', labels);
xlabel('Regime', 'FontSize', 15);
ylabel('Non-active classification risk', 'FontSize', 15);
legend({'Empirical','Oracle'}, 'Location', 'best', 'FontSize', 11);

exportgraphics(fig, fullfile(graph_dir, 'FINAL_ERROR_DECOMPOSITION.png'), 'Resolution', 300);
close(fig);

%% =========================================================
% FIGURE 5: Relative improvement over baseline versus rho_n
% ==========================================================
rows = [Cases.n] == 300 & [Cases.L] == 50;
cases_300_50 = Cases(rows);
[~, ord] = sort([cases_300_50.rho_n]);
cases_300_50 = cases_300_50(ord);

rho_vals = [cases_300_50.rho_n];
RI_emp = ([cases_300_50.calR_baseline] - [cases_300_50.calR_emp]) ./ [cases_300_50.calR_baseline];
RI_oracle = ([cases_300_50.calR_baseline] - [cases_300_50.calR_oracle]) ./ [cases_300_50.calR_baseline];

fig = figure('Color', 'w', 'Position', [100 100 900 500]);
hold on; box on;
plot(rho_vals, RI_emp, '-o', 'LineWidth', 2.4, 'MarkerSize', 7, 'Color', [0.88 0.45 0.12]);
plot(rho_vals, RI_oracle, '-s', 'LineWidth', 2.4, 'MarkerSize', 7, 'Color', [0.15 0.55 0.22]);
grid on;
disable_axes_toolbar(fig);
set(gca, 'FontSize', 14, 'LineWidth', 1.1);
xlabel('rho_n', 'FontSize', 17);
ylabel('Relative improvement over baseline', 'FontSize', 17);
legend({'Empirical','Oracle'}, 'Location', 'best', 'FontSize', 12);
if all([RI_emp, RI_oracle] >= 0)
    ylim([0, max([RI_emp, RI_oracle]) * 1.10 + eps]);
end
exportgraphics(fig, fullfile(graph_dir, 'FINAL_RELIMP_vs_rho.png'), 'Resolution', 300);
close(fig);

%% =========================================================
% TABLE 1: Threshold calibration table
% ==========================================================
table1_pref = [
    200   75   0.125
    250  100   0.150
    300   50   0.125
    350  125   0.100
    400  100   0.100
    450  150   0.050
];
table1_ids = find_closest_requested_cases(Cases, table1_pref);
table1_cases = Cases(table1_ids);

T1 = table( ...
    [table1_cases.n]', ...
    [table1_cases.L]', ...
    [table1_cases.rho_n]', ...
    [table1_cases.beta_hat_oracle]', ...
    round([table1_cases.beta_hat_emp]', 3), ...
    [table1_cases.calR_oracle]', ...
    [table1_cases.calR_emp]', ...
    'VariableNames', {'n','L','rho_n','beta_hat_oracle','beta_hat_emp','calR_oracle','calR_emp'});

writetable(T1, fullfile(table_dir, 'table_beta_calibration_layerB.csv'));
write_latex_table_beta(T1, fullfile(table_dir, 'table_beta_calibration_layerB.tex'));

%% =========================================================
% TABLE 2: Main comparison table
% ==========================================================
rows = {};
for i = 1:numel(fig3_cases)
    rows(end+1,:) = {fig3_cases(i).n, fig3_cases(i).L, fig3_cases(i).rho_n, 'Baseline', fig3_cases(i).calR_baseline}; %#ok<AGROW>
    rows(end+1,:) = {fig3_cases(i).n, fig3_cases(i).L, fig3_cases(i).rho_n, 'Empirical', fig3_cases(i).calR_emp}; %#ok<AGROW>
    rows(end+1,:) = {fig3_cases(i).n, fig3_cases(i).L, fig3_cases(i).rho_n, 'Oracle', fig3_cases(i).calR_oracle}; %#ok<AGROW>
end
T2 = cell2table(rows, 'VariableNames', {'n','L','rho_n','Method','calR_tot'});
writetable(T2, fullfile(table_dir, 'table_method_comparison_layerB.csv'));
write_latex_table_method(T2, fullfile(table_dir, 'table_method_comparison_layerB.tex'));

fprintf('\nSaved figures to:\n%s\n', graph_dir);
fprintf('Saved tables to:\n%s\n', table_dir);


%% =========================================================
% Local functions
% ==========================================================
function Cases = build_case_struct(T_oracle, T_compare, step2_dir)
nCases = height(T_oracle);
Cases = repmat(struct(), nCases, 1);

for i = 1:nCases
    n = T_oracle.n(i);
    L = T_oracle.L(i);
    rho_n = T_oracle.rho(i);

    mask = T_compare.n == n & T_compare.L == L & abs(T_compare.rho - rho_n) < 1e-12;
    if sum(mask) ~= 1
        error('STEP4:CaseMatch', 'Could not match case n=%d, L=%d, rho_n=%.3f.', n, L, rho_n);
    end

    row = T_compare(mask, :);
    Res = load_oracle_curve(step2_dir, n, L, T_oracle.d(i));

    grid_g = Res.beta_star_list(:)';
    err_curve = Res.err_curve(:)';
    [~, idx_emp_near] = min(abs(grid_g - row.betaEmpirical));

    Cases(i).n = n;
    Cases(i).L = L;
    Cases(i).d = T_oracle.d(i);
    Cases(i).rho_n = rho_n;
    Cases(i).grid_g = grid_g;
    Cases(i).beta_star_vec = Res.beta_vec;
    Cases(i).nL_vec = Res.nL_vec;
    Cases(i).err_curve = err_curve;
    Cases(i).active_count = Res.active_count(:)';
    Cases(i).beta_hat_oracle = row.betaOracle;
    Cases(i).beta_hat_emp = row.betaEmpirical;
    Cases(i).grid_emp = grid_g(idx_emp_near);
    Cases(i).idx_emp_near = idx_emp_near;
    Cases(i).calR_oracle = min(err_curve);
    Cases(i).calR_emp = err_curve(idx_emp_near);
    Cases(i).calR_baseline = row.errBaseline;
    Cases(i).delta_hat = row.deltaHat;
end
end

function confirm_checks(Cases)
for i = 1:numel(Cases)
    grid_g = Cases(i).grid_g;
    if ~any(abs(grid_g - Cases(i).beta_hat_oracle) < 1e-12)
        error('STEP4:OracleNotOnGrid', 'beta_hat_oracle is not a clean grid value.');
    end
    if abs(Cases(i).calR_oracle - min(Cases(i).err_curve)) > 1e-12
        error('STEP4:OracleRiskMismatch', 'calR_oracle is not the minimum grid risk.');
    end
    if abs(Cases(i).calR_emp - Cases(i).err_curve(Cases(i).idx_emp_near)) > 1e-12
        error('STEP4:EmpRiskMismatch', 'calR_emp is not using the nearest grid point.');
    end
end
fprintf('Final checks passed:\n');
fprintf('- beta_hat_oracle is grid-selected.\n');
fprintf('- beta_hat_emp is the raw regression value.\n');
fprintf('- calR_oracle uses the minimum grid risk.\n');
fprintf('- calR_emp uses the nearest-grid empirical risk.\n');
fprintf('- rho_n is used throughout plotting metadata.\n');
fprintf('- threshold plots use g on the x-axis.\n');
end

function ids = choose_distinct_cases_by_rho(Cases, n, L, rho_targets)
pool = find([Cases.n] == n & [Cases.L] == L);
if isempty(pool)
    error('STEP4:NoFixedCases', 'No cases found for n=%d, L=%d.', n, L);
end

ids = nan(1, numel(rho_targets));
used = false(size(pool));
for j = 1:numel(rho_targets)
    rho_pool = abs([Cases(pool).rho_n] - rho_targets(j));
    rho_pool(used) = inf;
    [~, k] = min(rho_pool);
    ids(j) = pool(k);
    used(k) = true;
end
end

function ids = find_exact_cases(Cases, target_mat)
ids = nan(1, size(target_mat, 1));
for i = 1:size(target_mat, 1)
    n = target_mat(i,1);
    L = target_mat(i,2);
    rho_n = target_mat(i,3);
    mask = [Cases.n] == n & [Cases.L] == L & abs([Cases.rho_n] - rho_n) < 1e-12;
    if sum(mask) == 1
        ids(i) = find(mask, 1);
    end
end
end

function ids = find_closest_requested_cases(Cases, target_mat)
ids = nan(1, size(target_mat, 1));
for i = 1:size(target_mat, 1)
    target = target_mat(i,:);
    score = abs([Cases.n] - target(1)) / max(target(1),1) + ...
            abs([Cases.L] - target(2)) / max(target(2),1) + ...
            5 * abs([Cases.rho_n] - target(3));
    [~, idx] = min(score);
    ids(i) = idx;
end
ids = unique_stable_ids(ids, Cases);
end

function ids = unique_stable_ids(ids, Cases)
used = false(1, numel(Cases));
for i = 1:numel(ids)
    if ~used(ids(i))
        used(ids(i)) = true;
        continue;
    end
    base = Cases(ids(i));
    score = abs([Cases.n] - base.n) / max(base.n,1) + ...
            abs([Cases.L] - base.L) / max(base.L,1) + ...
            5 * abs([Cases.rho_n] - base.rho_n);
    score(used) = inf;
    [~, idx] = min(score);
    ids(i) = idx;
    used(idx) = true;
end
end

function make_threshold_panels(case_list, outfile, xlab, ylab, title_fun, add_legend)
fig = figure('Color', 'w', 'Position', [100 100 1350 420]);
tiledlayout(1, numel(case_list), 'Padding', 'compact', 'TileSpacing', 'compact');
disable_axes_toolbar(fig);

ymin = inf;
ymax = -inf;
for i = 1:numel(case_list)
    ymin = min(ymin, min(case_list(i).err_curve));
    ymax = max(ymax, max(case_list(i).err_curve));
end

for i = 1:numel(case_list)
    S = case_list(i);
    nexttile; hold on; box on;
    plot(S.grid_g, S.err_curve, '-o', 'LineWidth', 2.3, 'MarkerSize', 6, 'Color', [0.10 0.35 0.70]);
    plot(S.beta_hat_oracle, S.calR_oracle, 'kx', 'LineWidth', 3, 'MarkerSize', 13);
    xline(S.beta_hat_emp, '--', 'LineWidth', 2, 'Color', [0.88 0.45 0.12]);
    grid on;
    set(gca, 'FontSize', 13, 'LineWidth', 1.1);
    xlabel(xlab, 'FontSize', 16);
    ylabel(ylab, 'FontSize', 16);
    ylim([max(0, ymin - 0.01), ymax + 0.02]);
    title(title_fun(S), 'FontSize', 15);
end

if add_legend
    lgd = legend({'Total risk', 'Oracle beta\_hat\_oracle', 'Empirical beta\_hat\_emp'}, ...
        'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 12);
    lgd.Layout.Tile = 'south';
end

exportgraphics(fig, outfile, 'Resolution', 300);
close(fig);
end

function add_bar_labels(b, vals)
for j = 1:numel(b)
    x = b(j).XEndPoints;
    y = b(j).YEndPoints;
    labs = compose('%.3f', vals(:,j));
    text(x, y + 0.01, labs, 'HorizontalAlignment', 'center', 'FontSize', 9);
end
end

function Res = load_oracle_curve(step2_dir, n, L, d)
fname = fullfile(step2_dir, sprintf('oracle_curve_n%d_L%d_d%03d.mat', n, L, round(100*d)));
if ~exist(fname, 'file')
    error('STEP4:MissingOracleCurve', 'Missing oracle curve file: %s', fname);
end
S = load(fname);
Res = S.Res;
end

function [emp_pair, oracle_pair] = recompute_decomposition(S, opts)
idx_oracle = nearest_grid_index(S.grid_g, S.beta_hat_oracle);
idx_emp = S.idx_emp_near;

beta_star_vec = S.beta_star_vec(:)';
g_oracle = S.grid_g(idx_oracle);
g_emp = S.grid_g(idx_emp);

idx_active_oracle = find(beta_star_vec >= g_oracle);
idx_nonactive_oracle = find(beta_star_vec < g_oracle);

idx_active_emp = find(beta_star_vec >= g_emp);
idx_nonactive_emp = find(beta_star_vec < g_emp);

act_emp = zeros(1, opts.Nruns);
non_emp = zeros(1, opts.Nruns);
act_oracle = zeros(1, opts.Nruns);
non_oracle = zeros(1, opts.Nruns);

for r = 1:opts.Nruns
    rng(r);
    [A, ~, ~, ~, ~, label] = ASBM_varNL_layerB( ...
        S.n, S.nL_vec, opts.K, S.L, opts.M, opts.c, S.d, opts.w);

    [~, act_emp(r), non_emp(r)] = split_error_decomp( ...
        A, label, idx_active_emp, idx_nonactive_emp, opts.M, opts.K, S.n, S.L);

    [~, act_oracle(r), non_oracle(r)] = split_error_decomp( ...
        A, label, idx_active_oracle, idx_nonactive_oracle, opts.M, opts.K, S.n, S.L);
end

emp_pair = [mean(act_emp), mean(non_emp)];
oracle_pair = [mean(act_oracle), mean(non_oracle)];
end

function idx = nearest_grid_index(grid_g, value)
[~, idx] = min(abs(grid_g - value));
end

function disable_axes_toolbar(fig)
axs = findall(fig, 'Type', 'axes');
for i = 1:numel(axs)
    try
        axs(i).Toolbar.Visible = 'off';
    catch
    end
end
end

function [calR_tot, calR_act, calR_non] = split_error_decomp(A, label, idx_active, idx_nonactive, M, K, n, L)
if isempty(idx_active) || numel(idx_active) < M
    calR_tot = 1;
    calR_act = 1;
    calR_non = 1;
    return;
end

A_active = A(idx_active);
[idcs_active, ~, ~] = BetweenLayerGramm_Gram(A_active, M, K);
if isempty(idcs_active) || any(isnan(idcs_active))
    calR_tot = 1;
    calR_act = 1;
    calR_non = 1;
    return;
end

s_hat = zeros(1, L);
s_hat(idx_active) = idcs_active(:)';

Uhat_group = cell(1, M);
for m = 1:M
    layers_m = idx_active(idcs_active == m);
    if isempty(layers_m)
        calR_tot = 1;
        calR_act = 1;
        calR_non = 1;
        return;
    end

    Hm = zeros(n, n);
    for ll = layers_m(:)'
        G = A{ll} * A{ll}';
        G(1:n+1:end) = 0;
        Hm = Hm + G;
    end

    try
        [Uhat, ~] = eigs(Hm, K, 'largestreal');
    catch
        calR_tot = 1;
        calR_act = 1;
        calR_non = 1;
        return;
    end
    Uhat_group{m} = Uhat;
end

for ll = idx_nonactive
    scores = zeros(1, M);
    for m = 1:M
        scores(m) = norm(Uhat_group{m}' * A{ll}, 'fro')^2;
    end
    [~, s_hat(ll)] = max(scores);
end

trueLab = label(:)';
miss_tot = missclassGroups_more(s_hat, trueLab, M);
calR_tot = miss_tot / L;

miss_act = missclassGroups_more(s_hat(idx_active), trueLab(idx_active), M);
calR_act = miss_act / numel(idx_active);

if isempty(idx_nonactive)
    calR_non = 0;
else
    miss_non = missclassGroups_more(s_hat(idx_nonactive), trueLab(idx_nonactive), M);
    calR_non = miss_non / numel(idx_nonactive);
end
end

function write_latex_table_beta(T, fname)
fid = fopen(fname, 'w');
fprintf(fid, '\\begin{tabular}{c c c c c c c}\n');
fprintf(fid, '\\hline\n');
fprintf(fid, '$n$ & $L$ & $\\rho_n$ & $\\hat{\\beta}_{\\mathrm{oracle}}$ & $\\hat{\\beta}_{\\mathrm{emp}}$ & $\\calR_{\\mathrm{oracle}}$ & $\\calR_{\\mathrm{emp}}$ \\\\\n');
fprintf(fid, '\\hline\n');
for i = 1:height(T)
    fprintf(fid, '%d & %d & %.3f & %.2f & %.3f & %.3f & %.3f \\\\\n', ...
        T.n(i), T.L(i), T.rho_n(i), T.beta_hat_oracle(i), T.beta_hat_emp(i), T.calR_oracle(i), T.calR_emp(i));
end
fprintf(fid, '\\hline\n');
fprintf(fid, '\\end{tabular}\n');
fclose(fid);
end

function write_latex_table_method(T, fname)
fid = fopen(fname, 'w');
fprintf(fid, '\\begin{tabular}{c c c l c}\n');
fprintf(fid, '\\hline\n');
fprintf(fid, '$n$ & $L$ & $\\rho_n$ & Method & $\\calR_{\\mathrm{tot}}$ \\\\\n');
fprintf(fid, '\\hline\n');
for i = 1:height(T)
    fprintf(fid, '%d & %d & %.3f & %s & %.3f \\\\\n', ...
        T.n(i), T.L(i), T.rho_n(i), T.Method{i}, T.calR_tot(i));
end
fprintf(fid, '\\hline\n');
fprintf(fid, '\\end{tabular}\n');
fclose(fid);
end
