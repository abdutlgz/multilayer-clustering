function run_method_validation_simulations(overwrite, numReps)
%RUN_METHOD_VALIDATION_SIMULATIONS Synthetic validation for the split pipeline.
%
% Focus:
%   - M = 2 and M = 3 only.
%   - Embedding figures rather than bar plots.
%   - Parameter sweeps that show when active-layer clustering plus
%     inactive-layer classification improves over all-layer clustering.
%
% Usage:
%   run_method_validation_simulations
%   run_method_validation_simulations(true, 30)

if nargin < 1 || isempty(overwrite)
    overwrite = false;
end
if nargin < 2 || isempty(numReps)
    numReps = 20;
end

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot)
    projectRoot = pwd;
end
addpath(projectRoot);

outDir = fullfile(projectRoot, 'simulations_method_validation');
resultDir = fullfile(outDir, 'results');
tableDir = fullfile(outDir, 'tables');
figureDir = fullfile(outDir, 'figures');
ensure_dir(outDir);
ensure_dir(resultDir);
ensure_dir(tableDir);
ensure_dir(figureDir);

summaryFile = fullfile(resultDir, 'method_validation_summary.mat');
if isfile(summaryFile) && ~overwrite
    fprintf('Loading existing simulation summary: %s\n', summaryFile);
    S = load(summaryFile);
    resultsTbl = S.resultsTbl;
    config = S.config;
else
    config = default_simulation_config(numReps);
    fprintf('Running synthetic method validation simulations...\n');
    fprintf('  numReps = %d\n', config.numReps);
    resultsTbl = run_validation_grid(config);
    save(summaryFile, 'resultsTbl', 'config', '-v7.3');
end

fprintf('Writing simulation tables...\n');
write_simulation_tables(resultsTbl, tableDir);

fprintf('Writing simulation figures...\n');
make_simulation_figures(resultsTbl, config, figureDir);

fprintf('Writing representative embeddings...\n');
make_representative_embeddings(config, figureDir, resultDir);

fprintf('Done. Synthetic validation outputs are in:\n');
fprintf('  %s\n', outDir);
end

function config = default_simulation_config(numReps)
config = struct();
config.numReps = numReps;
config.baseSeed = 20260514;
config.n = 240;
config.L = 64;
config.K = 3;
config.MList = [2 3];
config.betaStarList = [0.60 0.65 0.70 0.75 0.80 0.85];

config.heterogeneity = struct( ...
    'name', {'narrow_widths', 'movielens_like', 'wide_widths'}, ...
    'betaMin', {0.70, 0.55, 0.45}, ...
    'betaMax', {0.88, 0.95, 1.05});

config.signal = struct( ...
    'name', {'easy', 'medium', 'hard'}, ...
    'rho', {0.55, 0.42, 0.30}, ...
    'w', {0.15, 0.35, 0.55});

config.mainHeterogeneity = 'movielens_like';
config.mainSignal = 'medium';
config.mainBetaStar = 0.75;
end

function resultsTbl = run_validation_grid(config)
Role = cell(0,1);
Mcol = zeros(0,1);
Rep = zeros(0,1);
Heterogeneity = cell(0,1);
Signal = cell(0,1);
beta_star = zeros(0,1);
active_layers = zeros(0,1);
inactive_layers = zeros(0,1);
active_fraction = zeros(0,1);
proposed_error = zeros(0,1);
all_layer_error = zeros(0,1);
oracle_split_error = zeros(0,1);
inactive_margin = zeros(0,1);
status = cell(0,1);

row = 0;
for M = config.MList
    for h = 1:numel(config.heterogeneity)
        H = config.heterogeneity(h);
        for s = 1:numel(config.signal)
            Sig = config.signal(s);
            for rep = 1:config.numReps
                seed = config.baseSeed + 100000*M + 10000*h + 1000*s + rep;
                D = simulate_layer_dataset(config.n, config.L, config.K, M, ...
                    H.betaMin, H.betaMax, Sig.rho, Sig.w, seed);
                F = compute_layer_features(D.A_layers, config.K);
                allRes = cluster_all_layers(F.Theta_full, D.true_labels, M);

                for b = 1:numel(config.betaStarList)
                    betaStar = config.betaStarList(b);
                    row = row + 1;
                    Role{row,1} = 'grid'; %#ok<AGROW>
                    Mcol(row,1) = M; %#ok<AGROW>
                    Rep(row,1) = rep; %#ok<AGROW>
                    Heterogeneity{row,1} = H.name; %#ok<AGROW>
                    Signal{row,1} = Sig.name; %#ok<AGROW>
                    beta_star(row,1) = betaStar; %#ok<AGROW>

                    splitRes = run_split_pipeline(D.A_layers, F, D.true_labels, ...
                        D.beta_vec, betaStar, M, config.K);
                    oracleRes = run_oracle_split_classifier(D.A_layers, D.true_labels, ...
                        D.beta_vec, betaStar, M, config.K);

                    active_layers(row,1) = splitRes.numActive; %#ok<AGROW>
                    inactive_layers(row,1) = splitRes.numInactive; %#ok<AGROW>
                    active_fraction(row,1) = splitRes.numActive / config.L; %#ok<AGROW>
                    proposed_error(row,1) = splitRes.error; %#ok<AGROW>
                    all_layer_error(row,1) = allRes.error; %#ok<AGROW>
                    oracle_split_error(row,1) = oracleRes.error; %#ok<AGROW>
                    inactive_margin(row,1) = splitRes.meanInactiveMargin; %#ok<AGROW>
                    status{row,1} = splitRes.status; %#ok<AGROW>
                end
            end
        end
    end
end

resultsTbl = table(Role, Mcol, Rep, Heterogeneity, Signal, beta_star, ...
    active_layers, inactive_layers, active_fraction, proposed_error, ...
    all_layer_error, oracle_split_error, inactive_margin, status, ...
    'VariableNames', {'Role','M','Rep','Heterogeneity','Signal','beta_star', ...
    'Active_layers','Inactive_layers','Active_fraction','Proposed_error', ...
    'All_layer_error','Oracle_split_error','Inactive_margin','Status'});
end

function D = simulate_layer_dataset(n, L, K, M, betaMin, betaMax, rho, w, seed)
rng(seed);

beta_vec = betaMin + (betaMax - betaMin) * rand(1, L);
nL_vec = max(K + 2, round(n .^ beta_vec));

labels = repmat(1:M, 1, ceil(L / M));
labels = labels(1:L);
labels = labels(randperm(L));

ZX = cell(M,1);
for m = 1:M
    gx = randi(K, n, 1);
    ZX{m} = sparse(1:n, gx, 1, n, K);
end

A_layers = cell(1, L);
for ell = 1:L
    nl = nL_vec(ell);
    gy = randi(K, nl, 1);
    ZY = sparse(1:nl, gy, 1, nl, K);

    base = w * ones(K) + (1 - w) * eye(K);
    jitter = 0.85 + 0.30 * rand(K);
    B = rho * base .* jitter;
    B = min(B, 0.95);

    P = full(ZX{labels(ell)} * B * ZY');
    A_layers{ell} = sparse(rand(n, nl) < P);
end

D = struct();
D.A_layers = A_layers;
D.true_labels = labels(:);
D.beta_vec = beta_vec(:);
D.nL_vec = nL_vec(:);
end

function F = compute_layer_features(A_layers, K)
L = numel(A_layers);
n = size(A_layers{1}, 1);
U_all = cell(L,1);
featureMat = zeros(n * n, L);
J = eye(n) - ones(n) / n;

for ell = 1:L
    A = A_layers{ell};
    G = A * A';
    G(1:n+1:end) = 0;
    try
        [U,~] = eigs(G, K, 'largestreal');
    catch
        [Ufull,Dfull] = eig(full(G));
        [~,ord] = sort(diag(Dfull), 'descend');
        U = Ufull(:, ord(1:K));
    end
    U_all{ell} = U;
    H = J * U;
    featureMat(:,ell) = reshape(H * H', n * n, 1);
end

Theta = featureMat' * featureMat;
diagVals = sqrt(max(diag(Theta), eps));
Theta = Theta ./ (diagVals * diagVals');
Theta = (Theta + Theta') / 2;

F = struct();
F.U_all = U_all;
F.Theta_full = Theta;
end

function allRes = cluster_all_layers(Theta_full, trueLabels, M)
try
    embed = spectral_embed(Theta_full, M);
    pred = kmeans(embed, M, 'Replicates', 20, 'MaxIter', 1000);
    err = layer_misclassification(pred, trueLabels, M) / numel(trueLabels);
    status = 'ok';
catch ME
    pred = zeros(numel(trueLabels),1);
    err = 1;
    status = ME.identifier;
end

allRes = struct('pred', pred, 'error', err, 'status', status);
end

function splitRes = run_split_pipeline(A_layers, F, trueLabels, beta_vec, betaStar, M, K)
L = numel(A_layers);
activeIdx = find(beta_vec >= betaStar);
inactiveIdx = find(beta_vec < betaStar);

splitRes = empty_split_result(L, activeIdx, inactiveIdx);
if numel(activeIdx) < M
    splitRes.status = 'too_few_active';
    return;
end

try
    Theta_active = F.Theta_full(activeIdx, activeIdx);
    embedActive = spectral_embed(Theta_active, M);
    activeLabels = kmeans(embedActive, M, 'Replicates', 20, 'MaxIter', 1000);

    U_group = estimate_group_subspaces(A_layers, activeIdx, activeLabels, M, K);
    if any(cellfun(@isempty, U_group))
        splitRes.status = 'empty_estimated_cluster';
        return;
    end

    finalLabels = zeros(L,1);
    finalLabels(activeIdx) = activeLabels;
    [predInactive, margins] = classify_layers_to_groups(A_layers, inactiveIdx, U_group);
    finalLabels(inactiveIdx) = predInactive;

    splitRes.finalLabels = finalLabels;
    splitRes.activeLabels = activeLabels;
    splitRes.error = layer_misclassification(finalLabels, trueLabels, M) / L;
    splitRes.meanInactiveMargin = mean_or_nan(margins);
    splitRes.status = 'ok';
catch ME
    splitRes.status = ME.identifier;
end
end

function oracleRes = run_oracle_split_classifier(A_layers, trueLabels, beta_vec, betaStar, M, K)
L = numel(A_layers);
activeIdx = find(beta_vec >= betaStar);
inactiveIdx = find(beta_vec < betaStar);
oracleRes = empty_split_result(L, activeIdx, inactiveIdx);

if any(arrayfun(@(m) ~any(trueLabels(activeIdx) == m), 1:M))
    oracleRes.status = 'missing_true_group_active';
    return;
end

try
    U_group = estimate_group_subspaces(A_layers, activeIdx, trueLabels(activeIdx), M, K);
    finalLabels = zeros(L,1);
    finalLabels(activeIdx) = trueLabels(activeIdx);
    [predInactive, margins] = classify_layers_to_groups(A_layers, inactiveIdx, U_group);
    finalLabels(inactiveIdx) = predInactive;

    oracleRes.finalLabels = finalLabels;
    oracleRes.error = layer_misclassification(finalLabels, trueLabels, M) / L;
    oracleRes.meanInactiveMargin = mean_or_nan(margins);
    oracleRes.status = 'ok';
catch ME
    oracleRes.status = ME.identifier;
end
end

function R = empty_split_result(L, activeIdx, inactiveIdx)
R = struct();
R.finalLabels = zeros(L,1);
R.activeLabels = [];
R.numActive = numel(activeIdx);
R.numInactive = numel(inactiveIdx);
R.error = 1;
R.meanInactiveMargin = NaN;
R.status = 'not_run';
end

function U_group = estimate_group_subspaces(A_layers, layerIdx, layerLabels, M, K)
n = size(A_layers{1}, 1);
U_group = cell(M,1);
for m = 1:M
    idx = layerIdx(layerLabels(:) == m);
    if isempty(idx)
        U_group{m} = [];
        continue;
    end
    Hm = sparse(n, n);
    for ell = idx(:)'
        G = A_layers{ell} * A_layers{ell}';
        G(1:n+1:end) = 0;
        Hm = Hm + G;
    end
    try
        [U,~] = eigs(Hm, K, 'largestreal');
    catch
        [Ufull,Dfull] = eig(full(Hm));
        [~,ord] = sort(diag(Dfull), 'descend');
        U = Ufull(:, ord(1:K));
    end
    U_group{m} = U;
end
end

function [pred, margins] = classify_layers_to_groups(A_layers, layerIdx, U_group)
M = numel(U_group);
pred = zeros(numel(layerIdx),1);
margins = NaN(numel(layerIdx),1);
for t = 1:numel(layerIdx)
    ell = layerIdx(t);
    scores = zeros(1,M);
    for m = 1:M
        if isempty(U_group{m})
            scores(m) = -Inf;
        else
            scores(m) = norm(U_group{m}' * A_layers{ell}, 'fro')^2;
        end
    end
    [scoresSorted, ord] = sort(scores, 'descend');
    pred(t) = ord(1);
    if numel(scoresSorted) >= 2 && isfinite(scoresSorted(1)) && scoresSorted(1) > 0
        margins(t) = (scoresSorted(1) - scoresSorted(2)) / scoresSorted(1);
    end
end
end

function embed = spectral_embed(Theta, M)
Theta = (Theta + Theta') / 2;
if M >= size(Theta,1)
    [V,D] = eig(full(Theta));
    [~,ord] = sort(diag(D), 'descend');
    embed = V(:,ord(1:min(M,size(V,2))));
else
    try
        [embed,~] = eigs(Theta, M, 'largestreal');
    catch
        [V,D] = eig(full(Theta));
        [~,ord] = sort(diag(D), 'descend');
        embed = V(:,ord(1:M));
    end
end
rowNorms = sqrt(sum(embed.^2,2));
rowNorms(rowNorms == 0) = 1;
embed = embed ./ rowNorms;
end

function miss = layer_misclassification(predLabels, trueLabels, M)
predLabels = predLabels(:)';
trueLabels = trueLabels(:)';
permsM = perms(1:M);
missVals = zeros(size(permsM,1),1);
for p = 1:size(permsM,1)
    relabeled = permsM(p, trueLabels);
    missVals(p) = sum(predLabels ~= relabeled);
end
miss = min(missVals);
end

function write_simulation_tables(resultsTbl, tableDir)
writetable(resultsTbl, fullfile(tableDir, 'simulation_results_all.csv'));

G = groupsummary(resultsTbl, {'M','Heterogeneity','Signal','beta_star'}, ...
    {'mean','std'}, {'Active_fraction','Proposed_error','All_layer_error', ...
                     'Oracle_split_error','Inactive_margin'});
writetable(G, fullfile(tableDir, 'simulation_summary_by_setting.csv'));
write_latex_table_simple(first_n_rows(G, 40), fullfile(tableDir, 'simulation_summary_by_setting_preview.tex'));

mainMask = strcmp(resultsTbl.Heterogeneity, 'movielens_like') & ...
           strcmp(resultsTbl.Signal, 'medium') & ...
           abs(resultsTbl.beta_star - 0.75) < 1e-8;
Main = groupsummary(resultsTbl(mainMask,:), {'M'}, ...
    {'mean','std'}, {'Proposed_error','All_layer_error','Oracle_split_error', ...
                     'Active_fraction','Inactive_margin'});
writetable(Main, fullfile(tableDir, 'main_setting_M2_M3_comparison.csv'));
write_latex_table_simple(Main, fullfile(tableDir, 'main_setting_M2_M3_comparison.tex'));
end

function make_simulation_figures(resultsTbl, config, figureDir)
make_error_vs_beta_figure(resultsTbl, config, figureDir);
make_signal_sensitivity_figure(resultsTbl, config, figureDir);
make_margin_vs_beta_figure(resultsTbl, config, figureDir);
end

function make_error_vs_beta_figure(T, config, figureDir)
fig = figure('Color','w', 'Position', [100 100 1050 430]);
tiledlayout(1,2, 'TileSpacing','compact', 'Padding','compact');

for panel = 1:numel(config.MList)
    M = config.MList(panel);
    nexttile;
    hold on;
    colors = lines(numel(config.heterogeneity));
    for h = 1:numel(config.heterogeneity)
        hname = config.heterogeneity(h).name;
        mask = T.M == M & strcmp(T.Signal, config.mainSignal) & strcmp(T.Heterogeneity, hname);
        S = groupsummary(T(mask,:), 'beta_star', 'mean', {'Proposed_error','All_layer_error'});
        plot(S.beta_star, S.mean_Proposed_error, '-o', 'LineWidth', 1.8, ...
            'Color', colors(h,:), 'DisplayName', strrep(hname, '_', ' '));
        if h == 2
            plot(S.beta_star, S.mean_All_layer_error, '--', 'LineWidth', 1.5, ...
                'Color', [0.25 0.25 0.25], 'DisplayName', 'all-layer baseline');
        end
    end
    hold off;
    xlabel('\beta_*');
    ylabel('Mean misclassification error');
    title(sprintf('M = %d, medium signal', M));
    ylim([0 1]);
    grid on; box on;
    legend('Location','best');
    hide_toolbar(gca);
end

sgtitle('Proposed split pipeline across layer-width heterogeneity');
export_pair(fig, fullfile(figureDir, 'simulation_error_vs_beta_M2_M3'));
close(fig);
end

function make_signal_sensitivity_figure(T, config, figureDir)
fig = figure('Color','w', 'Position', [100 100 1050 430]);
tiledlayout(1,2, 'TileSpacing','compact', 'Padding','compact');
signalNames = {config.signal.name};
x = 1:numel(signalNames);

for panel = 1:numel(config.MList)
    M = config.MList(panel);
    nexttile;
    hold on;
    propMean = zeros(numel(signalNames),1);
    allMean = zeros(numel(signalNames),1);
    oracleMean = zeros(numel(signalNames),1);
    for s = 1:numel(signalNames)
        mask = T.M == M & strcmp(T.Heterogeneity, config.mainHeterogeneity) & ...
               strcmp(T.Signal, signalNames{s}) & abs(T.beta_star - config.mainBetaStar) < 1e-8;
        propMean(s) = mean(T.Proposed_error(mask), 'omitnan');
        allMean(s) = mean(T.All_layer_error(mask), 'omitnan');
        oracleMean(s) = mean(T.Oracle_split_error(mask), 'omitnan');
    end
    plot(x, propMean, '-o', 'LineWidth', 2, 'DisplayName', 'proposed');
    plot(x, allMean, '--s', 'LineWidth', 2, 'DisplayName', 'all-layer baseline');
    plot(x, oracleMean, ':d', 'LineWidth', 2, 'DisplayName', 'oracle split');
    hold off;
    xticks(x);
    xticklabels(strrep(signalNames, '_', ' '));
    xlabel('Signal regime');
    ylabel('Mean misclassification error');
    title(sprintf('M = %d, beta*=%.2f', M, config.mainBetaStar));
    ylim([0 1]);
    grid on; box on;
    legend('Location','best');
    hide_toolbar(gca);
end

sgtitle('Signal sensitivity');
export_pair(fig, fullfile(figureDir, 'simulation_signal_sensitivity_M2_M3'));
close(fig);
end

function make_margin_vs_beta_figure(T, config, figureDir)
fig = figure('Color','w', 'Position', [100 100 1050 430]);
tiledlayout(1,2, 'TileSpacing','compact', 'Padding','compact');

for panel = 1:numel(config.MList)
    M = config.MList(panel);
    nexttile;
    mask = T.M == M & strcmp(T.Heterogeneity, config.mainHeterogeneity) & strcmp(T.Signal, config.mainSignal);
    S = groupsummary(T(mask,:), 'beta_star', 'mean', {'Inactive_margin','Active_fraction'});
    yyaxis left;
    plot(S.beta_star, S.mean_Inactive_margin, '-o', 'LineWidth', 2);
    ylabel('Mean inactive classification margin');
    ylim([0 1]);
    yyaxis right;
    plot(S.beta_star, S.mean_Active_fraction, '--s', 'LineWidth', 2);
    ylabel('Mean active fraction');
    ylim([0 1]);
    xlabel('\beta_*');
    title(sprintf('M = %d, MovieLens-like widths', M));
    grid on; box on;
    hide_toolbar(gca);
end

sgtitle('Inactive classification confidence vs active-set size');
export_pair(fig, fullfile(figureDir, 'simulation_inactive_margin_vs_beta_M2_M3'));
close(fig);
end

function make_representative_embeddings(config, figureDir, resultDir)
for M = config.MList
    seed = config.baseSeed + 900000 + M;
    H = config.heterogeneity(strcmp({config.heterogeneity.name}, config.mainHeterogeneity));
    Sig = config.signal(strcmp({config.signal.name}, config.mainSignal));

    D = simulate_layer_dataset(config.n, config.L, config.K, M, ...
        H.betaMin, H.betaMax, Sig.rho, Sig.w, seed);
    F = compute_layer_features(D.A_layers, config.K);
    splitRes = run_split_pipeline(D.A_layers, F, D.true_labels, D.beta_vec, ...
        config.mainBetaStar, M, config.K);
    allRes = cluster_all_layers(F.Theta_full, D.true_labels, M);

    matFile = fullfile(resultDir, sprintf('representative_embedding_M%d.mat', M));
    save(matFile, 'D', 'F', 'splitRes', 'allRes', 'config', '-v7.3');

    plot_representative_embedding(F.Theta_full, D.true_labels, splitRes, allRes, ...
        D.beta_vec, config.mainBetaStar, M, figureDir);
end
end

function plot_representative_embedding(Theta_full, trueLabels, splitRes, allRes, beta_vec, betaStar, M, figureDir)
embed = spectral_embed(Theta_full, max(2, M));
activeIdx = find(beta_vec >= betaStar);
inactiveIdx = find(beta_vec < betaStar);
colors = lines(M);

fig = figure('Color','w', 'Position', [100 100 1050 430]);
tiledlayout(1,2, 'TileSpacing','compact', 'Padding','compact');

nexttile;
hold on;
for m = 1:M
    ia = intersect(activeIdx, find(splitRes.finalLabels == m));
    ii = intersect(inactiveIdx, find(splitRes.finalLabels == m));
    scatter(embed(ia,1), embed(ia,2), 70, colors(m,:), 'filled');
    scatter(embed(ii,1), embed(ii,2), 85, 'MarkerEdgeColor', colors(m,:), ...
        'MarkerFaceColor', 'none', 'LineWidth', 1.5);
end
bad = find_best_matched_errors(splitRes.finalLabels, trueLabels, M);
scatter(embed(bad,1), embed(bad,2), 120, 'kx', 'LineWidth', 1.7);
hold off;
title(sprintf('Proposed split pipeline, M=%d', M));
xlabel('Embedding coordinate 1');
ylabel('Embedding coordinate 2');
grid on; box on;
hide_toolbar(gca);

nexttile;
hold on;
for m = 1:M
    idx = find(allRes.pred == m);
    scatter(embed(idx,1), embed(idx,2), 70, colors(m,:), 'filled');
end
bad = find_best_matched_errors(allRes.pred, trueLabels, M);
scatter(embed(bad,1), embed(bad,2), 120, 'kx', 'LineWidth', 1.7);
hold off;
title(sprintf('All-layer baseline, M=%d', M));
xlabel('Embedding coordinate 1');
ylabel('Embedding coordinate 2');
grid on; box on;
hide_toolbar(gca);

sgtitle(sprintf('Representative synthetic embedding: beta*=%.2f, filled=active, hollow=inactive, x=mistake', betaStar));
export_pair(fig, fullfile(figureDir, sprintf('simulation_representative_embedding_M%d', M)));
close(fig);
end

function badIdx = find_best_matched_errors(predLabels, trueLabels, M)
predLabels = predLabels(:)';
trueLabels = trueLabels(:)';
permsM = perms(1:M);
bestMiss = Inf;
bestBad = [];
for p = 1:size(permsM,1)
    relabeled = permsM(p, trueLabels);
    bad = find(predLabels ~= relabeled);
    if numel(bad) < bestMiss
        bestMiss = numel(bad);
        bestBad = bad;
    end
end
badIdx = bestBad(:);
end

function Tsmall = first_n_rows(T, n)
Tsmall = T(1:min(n,height(T)), :);
end

function x = mean_or_nan(v)
if isempty(v)
    x = NaN;
else
    x = mean(v, 'omitnan');
end
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
