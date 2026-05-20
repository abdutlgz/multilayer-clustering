function run_paper_synthetic_algorithm_comparison(overwrite, numReps)
%RUN_PAPER_SYNTHETIC_ALGORITHM_COMPARISON Paper-aligned M=3 baseline study.
%
% This experiment extends the existing simulation section rather than
% replacing it.  It keeps the original M=3 rectangular SBM setup and the
% representative regimes used in the paper's baseline/empirical/oracle
% comparison, then adds HOOI and UASE baselines.
%
% Usage:
%   run_paper_synthetic_algorithm_comparison
%   run_paper_synthetic_algorithm_comparison(true, 50)

if nargin < 1 || isempty(overwrite)
    overwrite = false;
end
if nargin < 2 || isempty(numReps)
    numReps = 50;
end

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot)
    projectRoot = pwd;
end

addpath(projectRoot);
addpath(fullfile(projectRoot, 'legacy_unused_code', 'simulation_code'));
addpath(fullfile(projectRoot, 'legacy_unused_code', 'organized_simulations'));

outDir = fullfile(projectRoot, 'simulations_paper_algorithm_comparison');
resultDir = fullfile(outDir, 'results');
tableDir = fullfile(outDir, 'tables');
figureDir = fullfile(outDir, 'figures');
ensure_dir(outDir);
ensure_dir(resultDir);
ensure_dir(tableDir);
ensure_dir(figureDir);

summaryFile = fullfile(resultDir, 'paper_synthetic_algorithm_comparison.mat');
if isfile(summaryFile) && ~overwrite
    fprintf('Loading existing paper synthetic comparison: %s\n', summaryFile);
    S = load(summaryFile);
    resultTbl = S.resultTbl;
    config = S.config;
else
    config = default_paper_synthetic_config(numReps);
    fprintf('Running paper-aligned synthetic algorithm comparison\n');
    fprintf('  M = %d, K = %d, numReps = %d\n', config.M, config.K, config.numReps);
    resultTbl = run_paper_grid(config, projectRoot);
    save(summaryFile, 'resultTbl', 'config', '-v7.3');
end

fprintf('Writing paper synthetic comparison tables and figures...\n');
write_paper_synthetic_outputs(resultTbl, config, tableDir, figureDir);

fprintf('Done. Paper synthetic comparison outputs are in:\n');
fprintf('  %s\n', outDir);
end

function config = default_paper_synthetic_config(numReps)
config = struct();
config.numReps = numReps;
config.baseSeed = 20260519;
config.M = 3;
config.K = 3;
config.c = 0;
config.w = 1;
config.betaMin = 0.30;
config.betaMax = 1.10;

% These are the four regimes already displayed in the simulation section.
% Since c=0, rho_sim=d/2.
config.caseMat = [
    200   50   0.30
    300   50   0.25
    400  100   0.20
    450  150   0.10
];

config.methodOrder = {'Baseline_all_layer', 'Empirical_split', ...
    'Oracle_split', 'HOOI_gram_tensor', 'UASE_unfolded'};
end

function resultTbl = run_paper_grid(config, projectRoot)
calibrationTbl = load_delta_rule_table(projectRoot);

Case = cell(0,1);
nCol = zeros(0,1);
LCol = zeros(0,1);
dCol = zeros(0,1);
rhoCol = zeros(0,1);
Rep = zeros(0,1);
Method = cell(0,1);
Risk = zeros(0,1);
betaOracleCol = zeros(0,1);
betaEmpiricalCol = zeros(0,1);
ActiveOracle = zeros(0,1);
ActiveEmpirical = zeros(0,1);
Status = cell(0,1);

row = 0;
for i = 1:size(config.caseMat, 1)
    n = config.caseMat(i, 1);
    L = config.caseMat(i, 2);
    d = config.caseMat(i, 3);
    rho = (config.c + d) / 2;

    calibration = find_case_calibration(calibrationTbl, n, L, d);
    betaOracle = calibration.betaOracle;
    betaEmpirical = calibration.betaEmpirical;

    rng(config.baseSeed + 1000 * i);
    [nL_vec, beta_vec] = gen_nl_beta(n, L, config.betaMin, config.betaMax);
    nL_vec = max(nL_vec, config.K);
    beta_vec = log(nL_vec) ./ log(n);
    idxActiveOracle = find(beta_vec >= betaOracle);
    idxNonactiveOracle = find(beta_vec < betaOracle);
    idxActiveEmp = find(beta_vec >= betaEmpirical);
    idxNonactiveEmp = find(beta_vec < betaEmpirical);

    caseName = sprintf('n%d_L%d_rho%.3f', n, L, rho);
    fprintf('\nCase %d/%d: %s, d=%.2f\n', i, size(config.caseMat, 1), caseName, d);
    fprintf('  betaOracle=%.3f, betaEmpirical=%.3f, active empirical=%d/%d\n', ...
        betaOracle, betaEmpirical, numel(idxActiveEmp), L);

    for rep = 1:config.numReps
        if mod(rep, max(1, round(config.numReps / 10))) == 0 || rep == 1
            fprintf('  replicate %d/%d\n', rep, config.numReps);
        end

        rng(config.baseSeed + 100000 * i + rep);
        [A, ~, ~, ~, ~, label] = ASBM_varNL_layerB( ...
            n, nL_vec, config.K, L, config.M, config.c, d, config.w);

        [methodRisk, methodStatus] = evaluate_methods(A, label, ...
            idxActiveEmp, idxNonactiveEmp, idxActiveOracle, idxNonactiveOracle, config);

        for m = 1:numel(config.methodOrder)
            methodName = config.methodOrder{m};
            row = row + 1;
            Case{row,1} = caseName; %#ok<AGROW>
            nCol(row,1) = n; %#ok<AGROW>
            LCol(row,1) = L; %#ok<AGROW>
            dCol(row,1) = d; %#ok<AGROW>
            rhoCol(row,1) = rho; %#ok<AGROW>
            Rep(row,1) = rep; %#ok<AGROW>
            Method{row,1} = methodName; %#ok<AGROW>
            Risk(row,1) = methodRisk.(methodName); %#ok<AGROW>
            betaOracleCol(row,1) = betaOracle; %#ok<AGROW>
            betaEmpiricalCol(row,1) = betaEmpirical; %#ok<AGROW>
            ActiveOracle(row,1) = numel(idxActiveOracle); %#ok<AGROW>
            ActiveEmpirical(row,1) = numel(idxActiveEmp); %#ok<AGROW>
            Status{row,1} = methodStatus.(methodName); %#ok<AGROW>
        end
    end
end

resultTbl = table(Case, nCol, LCol, dCol, rhoCol, Rep, Method, Risk, ...
    betaOracleCol, betaEmpiricalCol, ActiveOracle, ActiveEmpirical, Status, ...
    'VariableNames', {'Case','n','L','d','rho','Rep','Method','Risk', ...
    'betaOracle','betaEmpirical','ActiveOracle','ActiveEmpirical','Status'});
end

function [methodRisk, methodStatus] = evaluate_methods(A, label, ...
    idxActiveEmp, idxNonactiveEmp, idxActiveOracle, idxNonactiveOracle, config)
methodRisk = struct();
methodStatus = struct();

[methodRisk.Baseline_all_layer, methodStatus.Baseline_all_layer] = ...
    safe_scalar_method(@() sim_baseline_error(A, label, config.M, config.K, numel(A)));

[methodRisk.Empirical_split, methodStatus.Empirical_split] = ...
    safe_scalar_method(@() sim_split_pipeline_error(A, label, idxActiveEmp, ...
    idxNonactiveEmp, config.M, config.K, size(A{1},1), numel(A)));

[methodRisk.Oracle_split, methodStatus.Oracle_split] = ...
    safe_scalar_method(@() sim_split_pipeline_error(A, label, idxActiveOracle, ...
    idxNonactiveOracle, config.M, config.K, size(A{1},1), numel(A)));

[methodRisk.HOOI_gram_tensor, methodStatus.HOOI_gram_tensor] = ...
    safe_struct_method(@() cluster_hooi_layers(A, label(:), config.M, config.K));

[methodRisk.UASE_unfolded, methodStatus.UASE_unfolded] = ...
    safe_struct_method(@() cluster_uase_layers(A, label(:), config.M, config.K));
end

function [risk, status] = safe_scalar_method(fun)
try
    risk = fun();
    if ~isfinite(risk)
        risk = 1;
        status = 'nonfinite';
    else
        status = 'ok';
    end
catch ME
    risk = 1;
    status = ME.identifier;
end
end

function [risk, status] = safe_struct_method(fun)
try
    S = fun();
    risk = S.error;
    status = S.status;
    if ~isfinite(risk)
        risk = 1;
        status = 'nonfinite';
    end
catch ME
    risk = 1;
    status = ME.identifier;
end
end

function calibrationTbl = load_delta_rule_table(projectRoot)
candidates = {
    fullfile(projectRoot, 'legacy_unused_code', 'organized_simulations', ...
    'organized_simulations', 'results', 'step3_delta_rule', 'delta_rule_comparison_table.csv')
    fullfile(projectRoot, 'legacy_unused_code', 'organized_simulations', ...
    'results', 'step3_delta_rule', 'delta_rule_comparison_table.csv')
};

calibrationTbl = [];
for i = 1:numel(candidates)
    if isfile(candidates{i})
        calibrationTbl = readtable(candidates{i});
        return;
    end
end
error('run_paper_synthetic_algorithm_comparison:MissingCalibration', ...
    'Could not find delta_rule_comparison_table.csv.');
end

function calibration = find_case_calibration(T, n, L, d)
idx = T.n == n & T.L == L & abs(T.d - d) < 1e-10;
if ~any(idx)
    error('run_paper_synthetic_algorithm_comparison:MissingCase', ...
        'No calibration row found for n=%d, L=%d, d=%.3f.', n, L, d);
end
row = find(idx, 1, 'first');
calibration = struct();
calibration.betaOracle = T.betaOracle(row);
calibration.betaEmpirical = T.betaEmpirical(row);
end

function write_paper_synthetic_outputs(resultTbl, config, tableDir, figureDir)
writetable(resultTbl, fullfile(tableDir, 'paper_synthetic_algorithm_comparison_all.csv'));

summaryTbl = groupsummary(resultTbl, {'Case','n','L','rho','Method'}, ...
    {'mean','std'}, 'Risk');
summaryTbl = order_summary_methods(summaryTbl, config.methodOrder);
writetable(summaryTbl, fullfile(tableDir, 'paper_synthetic_algorithm_comparison_summary.csv'));
write_latex_table_simple(summaryTbl, ...
    fullfile(tableDir, 'paper_synthetic_algorithm_comparison_summary.tex'));

compactTbl = make_compact_table(summaryTbl, config.methodOrder);
writetable(compactTbl, fullfile(tableDir, 'paper_synthetic_algorithm_comparison_compact.csv'));
write_latex_table_simple(compactTbl, ...
    fullfile(tableDir, 'paper_synthetic_algorithm_comparison_compact.tex'));

write_paper_synthetic_readme(summaryTbl, tableDir);
make_paper_synthetic_bar_figure(summaryTbl, config, figureDir);
end

function S = order_summary_methods(S, methodOrder)
caseNames = unique(string(S.Case), 'stable');
idxOrdered = [];
for c = 1:numel(caseNames)
    for m = 1:numel(methodOrder)
        idx = find(string(S.Case) == caseNames(c) & string(S.Method) == methodOrder{m});
        idxOrdered = [idxOrdered; idx(:)]; %#ok<AGROW>
    end
end
remaining = setdiff((1:height(S))', idxOrdered, 'stable');
S = S([idxOrdered; remaining],:);
end

function compactTbl = make_compact_table(S, methodOrder)
caseNames = unique(string(S.Case), 'stable');
Case = cell(numel(caseNames),1);
n = zeros(numel(caseNames),1);
L = zeros(numel(caseNames),1);
rho = zeros(numel(caseNames),1);
vals = NaN(numel(caseNames), numel(methodOrder));

for c = 1:numel(caseNames)
    Case{c} = char(caseNames(c));
    idxCase = string(S.Case) == caseNames(c);
    n(c) = S.n(find(idxCase, 1, 'first'));
    L(c) = S.L(find(idxCase, 1, 'first'));
    rho(c) = S.rho(find(idxCase, 1, 'first'));
    for m = 1:numel(methodOrder)
        idx = idxCase & string(S.Method) == methodOrder{m};
        if any(idx)
            vals(c,m) = S.mean_Risk(idx);
        end
    end
end

compactTbl = table(Case, n, L, rho, ...
    vals(:,1), vals(:,2), vals(:,3), vals(:,4), vals(:,5), ...
    'VariableNames', {'Case','n','L','rho', ...
    'Baseline_all_layer','Empirical_split','Oracle_split', ...
    'HOOI_gram_tensor','UASE_unfolded'});
end

function write_paper_synthetic_readme(S, tableDir)
outFile = fullfile(tableDir, 'paper_synthetic_algorithm_comparison_readme.txt');
fid = fopen(outFile, 'w');
if fid < 0
    error('Could not write %s', outFile);
end
cleaner = onCleanup(@() fclose(fid));

fprintf(fid, 'Paper-aligned synthetic algorithm comparison\n');
fprintf(fid, '===========================================\n\n');
fprintf(fid, 'This experiment keeps M=3 and the representative simulation regimes used in the paper.\n');
fprintf(fid, 'It extends the existing baseline/empirical/oracle table with HOOI and UASE baselines.\n\n');

caseNames = unique(string(S.Case), 'stable');
for c = 1:numel(caseNames)
    C = S(string(S.Case) == caseNames(c),:);
    fprintf(fid, '%s:\n', char(caseNames(c)));
    for r = 1:height(C)
        fprintf(fid, '  %s: mean risk=%.3f, sd=%.3f\n', ...
            char(C.Method{r}), C.mean_Risk(r), C.std_Risk(r));
    end
    fprintf(fid, '\n');
end
end

function make_paper_synthetic_bar_figure(S, config, figureDir)
caseNames = unique(string(S.Case), 'stable');
methods = config.methodOrder;
Y = NaN(numel(caseNames), numel(methods));
for c = 1:numel(caseNames)
    for m = 1:numel(methods)
        idx = string(S.Case) == caseNames(c) & string(S.Method) == methods{m};
        if any(idx)
            Y(c,m) = S.mean_Risk(idx);
        end
    end
end

fig = figure('Color','w', 'Position', [100 100 1100 440]);
bar(Y);
finiteY = Y(isfinite(Y));
if isempty(finiteY)
    yMax = 1;
else
    yMax = min(1, max(finiteY) * 1.25 + 0.02);
end
ylim([0, yMax]);
xticks(1:numel(caseNames));
xticklabels(strrep(caseNames, '_', '\_'));
xtickangle(20);
ylabel('Mean layer misclassification risk');
legend(strrep(methods, '_', ' '), 'Location', 'northoutside', ...
    'Orientation', 'horizontal');
title('M=3 synthetic algorithm comparison in paper regimes');
grid on; box on;
hide_toolbar(gca);
export_pair(fig, fullfile(figureDir, 'paper_synthetic_algorithm_comparison_bars'));
close(fig);
end

function export_pair(fig, outBase)
exportgraphics(fig, [outBase '.png'], 'Resolution', 300, 'BackgroundColor', 'white');
exportgraphics(fig, [outBase '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
end

function hide_toolbar(ax)
try
    ax.Toolbar.Visible = 'off';
catch
end
end

function ensure_dir(d)
if ~exist(d, 'dir')
    mkdir(d);
end
end
