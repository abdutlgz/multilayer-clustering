function run_movielens_realdata_scenarios(overwrite)
%RUN_MOVIELENS_REALDATA_SCENARIOS Real MovieLens stress tests for the paper.
%
% This is the paper-facing real-data scenario runner. It avoids broad beta
% sweeps and instead compares interpretable construction choices:
% users, number of layers, genre universe, positive-only signal, strict user
% activity, and equal-vs-uneven layer widths.
%
% Usage:
%   run_movielens_realdata_scenarios
%   run_movielens_realdata_scenarios(true)

if nargin < 1 || isempty(overwrite)
    overwrite = false;
end

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot)
    projectRoot = pwd;
end
addpath(projectRoot);
addpath(fullfile(projectRoot, 'ml-25m'));

dataFolder = fullfile(projectRoot, 'ml-25m');
outDir = fullfile(projectRoot, 'movielens_scenario_experiments');
buildDir = fullfile(outDir, 'builds');
resultDir = fullfile(outDir, 'results');
tableDir = fullfile(outDir, 'tables');
figureDir = fullfile(outDir, 'figures');
ensure_dir(outDir);
ensure_dir(buildDir);
ensure_dir(resultDir);
ensure_dir(tableDir);
ensure_dir(figureDir);

rngSeed = 7;
targetActiveFraction = 0.50;
MList = [2 3];
scenarios = movielens_scenario_design();

fprintf('MovieLens real-data scenario experiments\n');
fprintf('  overwrite = %d\n', overwrite);
fprintf('  scenarios = %d\n', numel(scenarios));
fprintf('  M values = [%s]\n', sprintf('%d ', MList));
fprintf('  active threshold rule = target %.0f%% active layers\n', 100 * targetActiveFraction);

needBuild = overwrite;
for s = 1:numel(scenarios)
    needBuild = needBuild || ~isfile(fullfile(buildDir, [scenarios(s).tag '.mat']));
end

ratingsTbl = [];
moviesTbl = [];
if needBuild
    fprintf('Loading MovieLens 25M tables...\n');
    [ratingsTbl, moviesTbl] = load_movielens25_tables(dataFolder);
end

runRows = cell(0, 13);
for s = 1:numel(scenarios)
    S = scenarios(s);
    buildFile = fullfile(buildDir, [S.tag '.mat']);

    if isfile(buildFile) && ~overwrite
        fprintf('Loading scenario build: %s\n', S.tag);
        B = load(buildFile);
        A_layers = B.A_layers;
        layer_info = ensure_density(B.layer_info);
        user_ids = B.user_ids;
    else
        if isempty(ratingsTbl) || isempty(moviesTbl)
            fprintf('Loading MovieLens 25M tables...\n');
            [ratingsTbl, moviesTbl] = load_movielens25_tables(dataFolder);
        end

        fprintf('Building scenario: %s\n', S.tag);
        [A_layers, layer_info, user_ids] = build_movielens_genre_splits( ...
            ratingsTbl, moviesTbl, ...
            S.minUserRatings, S.maxNumUsers, ...
            S.selectedGenres, S.numSplits, S.splitMode, rngSeed, ...
            S.usePositiveOnly, S.positiveThreshold);
        layer_info = ensure_density(layer_info);

        scenario = S; %#ok<NASGU>
        save(buildFile, 'A_layers', 'layer_info', 'user_ids', ...
            'scenario', 'rngSeed', '-v7.3');
    end

    needsRunWork = scenario_needs_run_work(resultDir, S.tag, MList, overwrite);
    if needsRunWork
        fprintf('Computing all-layer similarity for scenario: %s\n', S.tag);
        [Theta_full_base, ~, Khat_all] = compute_movielens_theta_for_scenario(A_layers);

        beta_star = choose_balanced_beta_star(layer_info.beta_l, targetActiveFraction);
        active_idx = find(layer_info.beta_l >= beta_star);
        inactive_idx = find(layer_info.beta_l < beta_star);
        fprintf('  beta*=%.4f gives %d active / %d non-active layers.\n', ...
            beta_star, numel(active_idx), numel(inactive_idx));
    else
        Theta_full_base = [];
        Khat_all = [];
        beta_star = NaN;
    end

    for M = MList
        runFile = fullfile(resultDir, sprintf('%s_M%d.mat', S.tag, M));
        if isfile(runFile) && ~overwrite
            R = load(runFile);
            final_labels = R.final_labels;
            V_all_embed = R.V_all_embed;
            active_idx = R.active_idx;
            inactive_idx = R.inactive_idx;
            beta_star = R.beta_star;
            inactive_results = R.inactive_results;
            layer_info = ensure_density(R.layer_info);
            scenario = R.scenario;
        else
            fprintf('  Running M=%d for scenario: %s\n', M, S.tag);
            R = run_scenario_pipeline(A_layers, layer_info, Theta_full_base, ...
                Khat_all, beta_star, M, rngSeed);

            scenario = S; %#ok<NASGU>
            active_idx = R.active_idx; %#ok<NASGU>
            inactive_idx = R.inactive_idx; %#ok<NASGU>
            active_results = R.active_results; %#ok<NASGU>
            inactive_results = R.inactive_results; %#ok<NASGU>
            final_labels = R.final_labels; %#ok<NASGU>
            Theta_hat = R.Theta_hat; %#ok<NASGU>
            cluster_labels = R.cluster_labels; %#ok<NASGU>
            U_group = R.U_group; %#ok<NASGU>
            Khat_group = R.Khat_group; %#ok<NASGU>
            Theta_full = Theta_full_base; %#ok<NASGU>
            V_all_embed = R.V_all_embed; %#ok<NASGU>

            save(runFile, 'scenario', 'rngSeed', 'beta_star', 'M', ...
                'A_layers', 'layer_info', 'user_ids', ...
                'active_idx', 'inactive_idx', 'active_results', ...
                'inactive_results', 'final_labels', 'Theta_hat', ...
                'cluster_labels', 'U_group', 'Khat_group', ...
                'Theta_full', 'V_all_embed', '-v7.3');
        end

        compTbl = movielens_cluster_composition(layer_info, final_labels, M, S.selectedGenres);
        compBase = fullfile(tableDir, sprintf('cluster_composition_%s_M%d', S.tag, M));
        writetable(compTbl, [compBase '.csv']);
        write_latex_table_simple(compTbl, [compBase '.tex']);

        make_scenario_embedding_figure(layer_info, V_all_embed, final_labels, active_idx, inactive_idx, ...
            S.tag, beta_star, M, figureDir);
        if M == 3 && size(V_all_embed, 2) >= 3
            make_scenario_embedding_pairplot(layer_info, V_all_embed, final_labels, active_idx, inactive_idx, ...
                S.tag, beta_star, M, figureDir);
        end

        runRows(end+1,:) = scenario_summary_row(S, layer_info, final_labels, ...
            active_idx, inactive_idx, inactive_results, V_all_embed, beta_star, M); %#ok<AGROW>
    end
end

summaryTbl = cell2table(runRows, 'VariableNames', { ...
    'Scenario','Purpose','M','Users','Genres','Splits_per_genre', ...
    'Total_layers','beta_star','Active_layers','Non_active_layers', ...
    'Genre_consistency','Embedding_silhouette','Inactive_margin'});
numericVars = {'M','Users','Genres','Splits_per_genre','Total_layers', ...
    'beta_star','Active_layers','Non_active_layers','Genre_consistency', ...
    'Embedding_silhouette','Inactive_margin'};
for v = 1:numel(numericVars)
    if iscell(summaryTbl.(numericVars{v}))
        summaryTbl.(numericVars{v}) = cell2mat(summaryTbl.(numericVars{v}));
    end
end
writetable(summaryTbl, fullfile(tableDir, 'scenario_run_summary.csv'));
write_latex_table_simple(summaryTbl, fullfile(tableDir, 'scenario_run_summary.tex'));

designTbl = scenario_design_table(scenarios);
writetable(designTbl, fullfile(tableDir, 'scenario_design.csv'));
write_latex_table_simple(designTbl, fullfile(tableDir, 'scenario_design.tex'));

make_embedding_panels(resultDir, scenarios, MList, figureDir);
make_scenario_metric_figure(summaryTbl, figureDir);
write_scenario_outcome_guide(summaryTbl, tableDir);

fprintf('Done. Scenario outputs are in:\n');
fprintf('  %s\n', outDir);
end

function tf = scenario_needs_run_work(resultDir, tag, MList, overwrite)
if overwrite
    tf = true;
    return;
end
for M = MList
    runFile = fullfile(resultDir, sprintf('%s_M%d.mat', tag, M));
    if ~isfile(runFile)
        tf = true;
        return;
    end
end
tf = false;
end

function [Theta_full, V_all_embed, Khat_all] = compute_movielens_theta_for_scenario(A_layers)
[Theta_full, V_all_embed, ~, Khat_all] = compute_movielens_theta_full(A_layers, ...
    'M', [], 'rank_method', 'energy', 'energy_thresh', 0.90, ...
    'fixed_rank', 5, 'maxRankTry', 10, 'use_normalized_similarity', true);
end

function scenarios = movielens_scenario_design()
core8 = {'Action','Adventure','Comedy','Crime','Drama','Romance','Sci-Fi','Thriller'};
genre12 = {'Action','Adventure','Animation','Comedy','Crime','Drama', ...
           'Fantasy','Horror','Mystery','Romance','Sci-Fi','Thriller'};

base = struct('tag', '', 'purpose', '', 'minUserRatings', 40, ...
    'maxNumUsers', 8000, 'selectedGenres', {core8}, 'numSplits', 8, ...
    'splitMode', 'uneven', 'usePositiveOnly', false, 'positiveThreshold', 4.0);

scenarios = repmat(base, 1, 9);

scenarios(1) = set_scenario(base, 'main_u8000_core8_s8', ...
    'Main MovieLens setting: 8000 users, 8 core genres, 64 layers.', ...
    40, 8000, core8, 8, 'uneven', false, 4.0);

scenarios(2) = set_scenario(base, 'users3000_core8_s8', ...
    'Fewer users: tests whether structure is visible with less row information.', ...
    40, 3000, core8, 8, 'uneven', false, 4.0);

scenarios(3) = set_scenario(base, 'users12000_core8_s8', ...
    'More users: tests whether embeddings sharpen with more row information.', ...
    40, 12000, core8, 8, 'uneven', false, 4.0);

scenarios(4) = set_scenario(base, 'layers32_core8_s4', ...
    'Fewer wider layers: 8 genres x 4 splits.', ...
    40, 8000, core8, 4, 'uneven', false, 4.0);

scenarios(5) = set_scenario(base, 'layers96_core8_s12', ...
    'More thinner layers: 8 genres x 12 splits.', ...
    40, 8000, core8, 12, 'uneven', false, 4.0);

scenarios(6) = set_scenario(base, 'genres12_u8000_s6', ...
    'Broader genre universe: 12 genres x 6 splits.', ...
    40, 8000, genre12, 6, 'uneven', false, 4.0);

scenarios(7) = set_scenario(base, 'strictusers_u8000_min80_core8_s8', ...
    'More active retained users: min 80 ratings before selecting users.', ...
    80, 8000, core8, 8, 'uneven', false, 4.0);

scenarios(8) = set_scenario(base, 'positive4_u8000_core8_s8', ...
    'Positive-preference signal: only ratings >= 4.', ...
    40, 8000, core8, 8, 'uneven', true, 4.0);

scenarios(9) = set_scenario(base, 'equalwidth_u8000_core8_s8', ...
    'Equal-width control: checks whether width heterogeneity drives the picture.', ...
    40, 8000, core8, 8, 'equal', false, 4.0);
end

function S = set_scenario(base, tag, purpose, minUserRatings, maxNumUsers, selectedGenres, numSplits, splitMode, usePositiveOnly, positiveThreshold)
S = base;
S.tag = tag;
S.purpose = purpose;
S.minUserRatings = minUserRatings;
S.maxNumUsers = maxNumUsers;
S.selectedGenres = selectedGenres;
S.numSplits = numSplits;
S.splitMode = splitMode;
S.usePositiveOnly = usePositiveOnly;
S.positiveThreshold = positiveThreshold;
end

function R = run_scenario_pipeline(A_layers, layer_info, Theta_full, Khat_all, beta_star, M, rngSeed)
rng(rngSeed);
active_idx = find(layer_info.beta_l >= beta_star);
inactive_idx = find(layer_info.beta_l < beta_star);

if numel(active_idx) < M
    error('Scenario has only %d active layers for M=%d.', numel(active_idx), M);
end

Theta_hat = Theta_full(active_idx, active_idx);
V_active = spectral_embed_from_similarity(Theta_hat, M);
cluster_labels = kmeans(V_active, M, 'Replicates', 40, 'MaxIter', 1000);

U_group = cell(M,1);
Khat_group = zeros(M,1);
A_active = A_layers(active_idx);
for m = 1:M
    idx_m = find(cluster_labels == m);
    if isempty(idx_m)
        U_group{m} = [];
        Khat_group(m) = 0;
        continue;
    end
    K_group = max(1, round(median(Khat_all(active_idx(idx_m)))));
    Khat_group(m) = K_group;
    A_concat = horzcat(A_active{idx_m});
    U_group{m} = left_svd_subspace(A_concat, K_group);
end

if ~isempty(inactive_idx)
    inactive_results = classify_inactive_layers_genre(A_layers, layer_info, inactive_idx, U_group);
else
    inactive_results = table();
end

active_results = layer_info(active_idx,:);
active_results.original_layer_id = active_idx(:);
active_results.Khat = Khat_all(active_idx);
active_results.cluster = cluster_labels(:);

final_labels = get_movielens_final_labels(numel(A_layers), active_idx, ...
    cluster_labels, inactive_idx, inactive_results);
V_all_embed = spectral_embed_from_similarity(Theta_full, max(2, M));

R = struct();
R.active_idx = active_idx;
R.inactive_idx = inactive_idx;
R.active_results = active_results;
R.inactive_results = inactive_results;
R.final_labels = final_labels;
R.Theta_hat = Theta_hat;
R.cluster_labels = cluster_labels;
R.U_group = U_group;
R.Khat_group = Khat_group;
R.V_all_embed = V_all_embed;
end

function beta_star = choose_balanced_beta_star(betaVals, targetActiveFraction)
betaVals = sort(betaVals(:), 'descend');
L = numel(betaVals);
targetActive = round(targetActiveFraction * L);
targetActive = min(max(targetActive, 2), L - 1);

if targetActive < L
    beta_star = 0.5 * (betaVals(targetActive) + betaVals(targetActive + 1));
else
    beta_star = betaVals(end);
end
end

function make_scenario_embedding_figure(layer_info, V, labels, active_idx, inactive_idx, tag, beta_star, M, figureDir)
colors = lines(M);
fig = figure('Color','w', 'Position', [100 100 690 560]);
hold on;
for m = 1:M
    idxA = intersect(active_idx(:), find(labels == m));
    idxI = intersect(inactive_idx(:), find(labels == m));
    scatter(V(idxA,1), V(idxA,2), 72, colors(m,:), 'filled', 'o');
    scatter(V(idxI,1), V(idxI,2), 90, 'MarkerEdgeColor', colors(m,:), ...
        'MarkerFaceColor', 'none', 'Marker', 'o', 'LineWidth', 1.5);
end
add_genre_labels(layer_info, V);
hold off;
xlabel('Embedding coordinate 1');
ylabel('Embedding coordinate 2');
title(sprintf('%s, M=%d, beta*=%.3f', tag, M, beta_star), 'Interpreter', 'none');
grid on; box on;
hide_toolbar(gca);
export_pair(fig, fullfile(figureDir, sprintf('embedding_%s_M%d', tag, M)));
close(fig);
end

function make_scenario_embedding_pairplot(layer_info, V, labels, active_idx, inactive_idx, tag, beta_star, M, figureDir)
pairs = [1 2; 1 3; 2 3];
colors = lines(M);
fig = figure('Color','w', 'Position', [80 80 1320 420]);
tiledlayout(1,3, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:size(pairs,1)
    nexttile;
    hold on;
    xcoord = pairs(p,1);
    ycoord = pairs(p,2);
    for m = 1:M
        idxA = intersect(active_idx(:), find(labels == m));
        idxI = intersect(inactive_idx(:), find(labels == m));
        scatter(V(idxA,xcoord), V(idxA,ycoord), 44, colors(m,:), 'filled', 'o');
        scatter(V(idxI,xcoord), V(idxI,ycoord), 52, 'MarkerEdgeColor', colors(m,:), ...
            'MarkerFaceColor', 'none', 'Marker', 'o', 'LineWidth', 1.2);
    end
    add_genre_labels_pair(layer_info, V, xcoord, ycoord);
    hold off;
    xlabel(sprintf('Embedding coordinate %d', xcoord));
    ylabel(sprintf('Embedding coordinate %d', ycoord));
    title(sprintf('coords %d-%d', xcoord, ycoord));
    grid on; box on;
    hide_toolbar(gca);
end
sgtitle(sprintf('%s, M=%d, beta*=%.3f', tag, M, beta_star), 'Interpreter', 'none');
export_pair(fig, fullfile(figureDir, sprintf('embedding_paircoords_%s_M%d', tag, M)));
close(fig);
end

function add_genre_labels(layer_info, V)
genres = unique(string(layer_info.genre), 'stable');
for g = 1:numel(genres)
    idx = find(string(layer_info.genre) == genres(g));
    if isempty(idx)
        continue;
    end
    xy = median(V(idx,1:2), 1);
    text(xy(1), xy(2), char(genres(g)), 'FontSize', 8, ...
        'HorizontalAlignment', 'center', 'BackgroundColor', 'w', ...
        'Margin', 1, 'Interpreter', 'none');
end
end

function add_genre_labels_pair(layer_info, V, xcoord, ycoord)
genres = unique(string(layer_info.genre), 'stable');
for g = 1:numel(genres)
    idx = find(string(layer_info.genre) == genres(g));
    if isempty(idx)
        continue;
    end
    xy = median(V(idx,[xcoord ycoord]), 1);
    text(xy(1), xy(2), char(genres(g)), 'FontSize', 7, ...
        'HorizontalAlignment', 'center', 'BackgroundColor', 'w', ...
        'Margin', 1, 'Interpreter', 'none');
end
end

function row = scenario_summary_row(S, layer_info, final_labels, active_idx, inactive_idx, inactive_results, V, beta_star, M)
genreConsistency = genre_consistency_score(layer_info, final_labels, M);
sil = embedding_silhouette_score(V(:,1:2), final_labels);
inactiveMargin = inactive_margin_score(inactive_results, M);
row = {S.tag, S.purpose, M, layer_info.n_users(1), numel(S.selectedGenres), ...
    S.numSplits, height(layer_info), beta_star, numel(active_idx), ...
    numel(inactive_idx), genreConsistency, sil, inactiveMargin};
end

function margin = inactive_margin_score(inactive_results, M)
if isempty(inactive_results)
    margin = NaN;
    return;
end
scoreMat = NaN(height(inactive_results), M);
for m = 1:M
    varName = sprintf('score_group_%d', m);
    if ismember(varName, inactive_results.Properties.VariableNames)
        scoreMat(:,m) = inactive_results.(varName);
    end
end
if all(isnan(scoreMat(:)))
    margin = NaN;
    return;
end
scoreMat(~isfinite(scoreMat)) = NaN;
margins = NaN(size(scoreMat,1),1);
for i = 1:size(scoreMat,1)
    s = scoreMat(i,:);
    s = s(isfinite(s));
    s = sort(s, 'descend');
    if numel(s) >= 2 && isfinite(s(1)) && isfinite(s(2)) && s(1) > 0
        margins(i) = (s(1) - s(2)) / s(1);
    end
end
margin = mean_omitnan(margins);
end

function score = genre_consistency_score(layer_info, final_labels, M)
genres = unique(string(layer_info.genre), 'stable');
vals = zeros(numel(genres),1);
for g = 1:numel(genres)
    idx = string(layer_info.genre) == genres(g);
    counts = zeros(1,M);
    for m = 1:M
        counts(m) = sum(idx & final_labels(:) == m);
    end
    vals(g) = max(counts) / sum(idx);
end
score = mean(vals);
end

function score = embedding_silhouette_score(X, labels)
labels = labels(:);
n = size(X,1);
if numel(unique(labels)) < 2
    score = NaN;
    return;
end
D = pairwise_distances(X);
s = NaN(n,1);
for i = 1:n
    own = labels == labels(i);
    own(i) = false;
    if any(own)
        a = mean(D(i,own));
    else
        a = 0;
    end
    otherLabels = setdiff(unique(labels), labels(i));
    bVals = zeros(numel(otherLabels),1);
    for k = 1:numel(otherLabels)
        bVals(k) = mean(D(i, labels == otherLabels(k)));
    end
    b = min(bVals);
    s(i) = (b - a) / max(a, b);
end
score = mean_omitnan(s);
end

function D = pairwise_distances(X)
n = size(X,1);
D = zeros(n,n);
for i = 1:n
    diffs = X - X(i,:);
    D(i,:) = sqrt(sum(diffs.^2, 2));
end
end

function x = mean_omitnan(v)
v = v(isfinite(v));
if isempty(v)
    x = NaN;
else
    x = mean(v);
end
end

function make_embedding_panels(resultDir, scenarios, MList, figureDir)
for M = MList
    fig = figure('Color','w', 'Position', [80 80 1180 920]);
    tiledlayout(3,3, 'TileSpacing', 'compact', 'Padding', 'compact');
    for s = 1:numel(scenarios)
        S = scenarios(s);
        R = load(fullfile(resultDir, sprintf('%s_M%d.mat', S.tag, M)), ...
            'V_all_embed', 'final_labels', 'active_idx', 'inactive_idx', 'beta_star');
        nexttile;
        colors = lines(M);
        hold on;
        for m = 1:M
            idxA = intersect(R.active_idx(:), find(R.final_labels == m));
            idxI = intersect(R.inactive_idx(:), find(R.final_labels == m));
            scatter(R.V_all_embed(idxA,1), R.V_all_embed(idxA,2), 25, colors(m,:), 'filled');
            scatter(R.V_all_embed(idxI,1), R.V_all_embed(idxI,2), 32, ...
                'MarkerEdgeColor', colors(m,:), 'MarkerFaceColor', 'none', 'LineWidth', 1.0);
        end
        hold off;
        title(sprintf('%s (beta=%.3f)', S.tag, R.beta_star), 'Interpreter', 'none', 'FontSize', 8);
        axis tight;
        grid on; box on;
        hide_toolbar(gca);
    end
    sgtitle(sprintf('MovieLens scenario embeddings, M=%d', M));
    export_pair(fig, fullfile(figureDir, sprintf('embedding_panel_all_scenarios_M%d', M)));
    close(fig);
end
end

function make_scenario_metric_figure(T, figureDir)
scens = unique(string(T.Scenario), 'stable');
fig = figure('Color','w', 'Position', [100 100 1320 500]);
tiledlayout(1,3, 'TileSpacing', 'compact', 'Padding', 'compact');

metrics = {'Genre_consistency','Embedding_silhouette','Inactive_margin'};
titles = {'Genre-split consistency', 'Embedding silhouette', 'Inactive assignment margin'};
for p = 1:numel(metrics)
    nexttile;
    hold on;
    for M = [2 3]
        vals = NaN(numel(scens),1);
        for i = 1:numel(scens)
            idx = string(T.Scenario) == scens(i) & T.M == M;
            vals(i) = T.(metrics{p})(idx);
        end
        plot(1:numel(scens), vals, '-o', 'LineWidth', 1.7, ...
            'DisplayName', sprintf('M=%d', M));
    end
    hold off;
    xticks(1:numel(scens));
    xticklabels(scens);
    xtickangle(35);
    ylabel('Score');
    title(titles{p});
    ylim([-0.1 1.05]);
    grid on; box on;
    legend('Location', 'best');
    hide_toolbar(gca);
end

sgtitle('Scenario-level stability metrics');
export_pair(fig, fullfile(figureDir, 'scenario_stability_metrics_M2_M3'));
close(fig);
end

function write_scenario_outcome_guide(T, tableDir)
outFile = fullfile(tableDir, 'scenario_experiment_readme.txt');
fid = fopen(outFile, 'w');
if fid < 0
    error('Could not write %s', outFile);
end
cleaner = onCleanup(@() fclose(fid));

fprintf(fid, 'MovieLens real-data scenario experiments\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'This experiment varies real construction parameters rather than beta grids.\n');
fprintf(fid, 'Each scenario uses an adaptive beta_star chosen to keep about half of the layers active and half non-active.\n');
fprintf(fid, 'The main figures are embeddings for M=2 and M=3, including active/non-active markers.\n\n');

for M = [2 3]
    fprintf(fid, 'M = %d scenario summary:\n', M);
    S = T(T.M == M,:);
    marginForScore = S.Inactive_margin;
    marginForScore(isnan(marginForScore)) = 0;
    totalScore = S.Genre_consistency + S.Embedding_silhouette + marginForScore;
    [~,ord] = sort(totalScore, 'descend');
    S = S(ord,:);
    for i = 1:height(S)
        fprintf(fid, '  %s: active=%d, inactive=%d, genre consistency=%.3f, silhouette=%.3f, inactive margin=%.3f\n', ...
            char(string(S.Scenario{i})), S.Active_layers(i), S.Non_active_layers(i), ...
            S.Genre_consistency(i), S.Embedding_silhouette(i), S.Inactive_margin(i));
    end
    fprintf(fid, '\n');
end

fprintf(fid, 'Suggested paper figures:\n');
fprintf(fid, '  figures/embedding_panel_all_scenarios_M2.pdf\n');
fprintf(fid, '  figures/embedding_panel_all_scenarios_M3.pdf\n');
fprintf(fid, '  figures/embedding_main_u8000_core8_s8_M2.pdf\n');
fprintf(fid, '  figures/embedding_main_u8000_core8_s8_M3.pdf\n');
fprintf(fid, '  figures/scenario_stability_metrics_M2_M3.pdf\n');
end

function T = scenario_design_table(scenarios)
Scenario = cell(numel(scenarios),1);
Purpose = cell(numel(scenarios),1);
Min_user_ratings = zeros(numel(scenarios),1);
Max_users = zeros(numel(scenarios),1);
Genres = zeros(numel(scenarios),1);
Splits_per_genre = zeros(numel(scenarios),1);
Split_mode = cell(numel(scenarios),1);
Positive_only = false(numel(scenarios),1);

for i = 1:numel(scenarios)
    S = scenarios(i);
    Scenario{i} = S.tag;
    Purpose{i} = S.purpose;
    Min_user_ratings(i) = S.minUserRatings;
    Max_users(i) = S.maxNumUsers;
    Genres(i) = numel(S.selectedGenres);
    Splits_per_genre(i) = S.numSplits;
    Split_mode{i} = S.splitMode;
    Positive_only(i) = S.usePositiveOnly;
end

T = table(Scenario, Purpose, Min_user_ratings, Max_users, Genres, ...
    Splits_per_genre, Split_mode, Positive_only);
end

function V_embed = spectral_embed_from_similarity(Theta, M)
Theta = (Theta + Theta') / 2;
if M >= size(Theta,1)
    [V,D] = eig(full(Theta));
    [~,ord] = sort(diag(D), 'descend');
    V_embed = V(:, ord(1:min(M,size(V,2))));
else
    try
        [V_embed,~] = eigs(Theta, M, 'largestreal');
    catch
        [V,D] = eig(full(Theta));
        [~,ord] = sort(diag(D), 'descend');
        V_embed = V(:,ord(1:M));
    end
end
rowNorms = sqrt(sum(V_embed.^2, 2));
rowNorms(rowNorms == 0) = 1;
V_embed = V_embed ./ rowNorms;
end

function U = left_svd_subspace(A, K)
minDim = min(size(A));
K = max(1, min(K, minDim));
if K >= minDim
    if minDim <= 500
        [Ufull,~,~] = svd(full(A), 'econ');
        U = Ufull(:,1:K);
        return;
    end
    K = minDim - 1;
end
[U,~,~] = svds(A, K);
end

function layer_info = ensure_density(layer_info)
if ~ismember('density', layer_info.Properties.VariableNames)
    layer_info.density = layer_info.n_edges ./ (layer_info.n_users .* layer_info.n_movies);
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
