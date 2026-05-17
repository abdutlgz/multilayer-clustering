function run_movielens_split_advantage_scenarios(overwrite)
%RUN_MOVIELENS_SPLIT_ADVANTAGE_SCENARIOS MovieLens stress tests for split advantage.
%
% This experiment creates real-data constructions where the split method
% should be useful: each genre has a few wide/high-signal anchor layers and
% many thin/edge-subsampled weak layers. The active set is chosen by layer
% width, so active layers estimate structure and inactive layers test
% classification.
%
% Usage:
%   run_movielens_split_advantage_scenarios
%   run_movielens_split_advantage_scenarios(true)

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
outDir = fullfile(projectRoot, 'movielens_split_advantage_experiments');
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
MList = [2 3];
scenarios = split_advantage_design();

fprintf('MovieLens split-advantage stress tests\n');
fprintf('  overwrite = %d\n', overwrite);
fprintf('  scenarios = %d\n', numel(scenarios));

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

summaryRows = cell(0, 13);
for s = 1:numel(scenarios)
    S = scenarios(s);
    buildFile = fullfile(buildDir, [S.tag '.mat']);

    if isfile(buildFile) && ~overwrite
        fprintf('Loading stress build: %s\n', S.tag);
        B = load(buildFile);
        A_layers = B.A_layers;
        layer_info = ensure_density(B.layer_info);
        user_ids = B.user_ids;
    else
        if isempty(ratingsTbl) || isempty(moviesTbl)
            fprintf('Loading MovieLens 25M tables...\n');
            [ratingsTbl, moviesTbl] = load_movielens25_tables(dataFolder);
        end

        fprintf('Building stress scenario: %s\n', S.tag);
        [A_layers, layer_info, user_ids] = build_mixed_signal_layers(ratingsTbl, moviesTbl, S, rngSeed);
        layer_info = ensure_density(layer_info);
        scenario = S; %#ok<NASGU>
        save(buildFile, 'A_layers', 'layer_info', 'user_ids', 'scenario', 'rngSeed', '-v7.3');
    end

    needsRunWork = stress_scenario_needs_run_work(resultDir, S.tag, MList, overwrite);
    if needsRunWork
        fprintf('Computing all-layer similarity: %s\n', S.tag);
        [Theta_full, ~, ~, Khat_all] = compute_movielens_theta_full(A_layers, ...
            'M', [], 'rank_method', 'energy', 'energy_thresh', 0.90, ...
            'fixed_rank', 5, 'maxRankTry', 10, 'use_normalized_similarity', true);
    else
        Theta_full = [];
        Khat_all = [];
    end

    for M = MList
        runFile = fullfile(resultDir, sprintf('%s_M%d.mat', S.tag, M));
        if isfile(runFile) && ~overwrite
            R = load(runFile);
        else
            fprintf('  Running algorithms for %s, M=%d\n', S.tag, M);
            R = run_stress_algorithms(A_layers, layer_info, Theta_full, Khat_all, M, rngSeed);
            scenario = S; %#ok<NASGU>
            save(runFile, '-struct', 'R', '-v7.3');
            save(runFile, 'scenario', 'rngSeed', '-append');
        end

        algNames = fieldnames(R.alg_labels);
        for a = 1:numel(algNames)
            algName = algNames{a};
            labels = R.alg_labels.(algName);
            compTbl = movielens_cluster_composition(layer_info, labels, M, S.selectedGenres);
            compBase = fullfile(tableDir, sprintf('composition_%s_M%d_%s', S.tag, M, algName));
            writetable(compTbl, [compBase '.csv']);
            write_latex_table_simple(compTbl, [compBase '.tex']);

            metrics = stress_metrics(layer_info, labels, R.active_idx, R.inactive_idx, M, R.V_all_embed);
            summaryRows(end+1,:) = {S.tag, S.purpose, M, algName, height(layer_info), ...
                numel(R.active_idx), numel(R.inactive_idx), R.beta_star, ...
                metrics.genreConsistency, metrics.weakConsistency, ...
                metrics.weakAnchorAgreement, metrics.embeddingSilhouette, ...
                label_agreement(R.alg_labels.proposed_split_subspace, labels, M)}; %#ok<AGROW>
        end

        make_stress_algorithm_embedding_figure(layer_info, R, M, S.tag, figureDir);
    end
end

summaryTbl = cell2table(summaryRows, 'VariableNames', { ...
    'Scenario','Purpose','M','Algorithm','Total_layers','Active_layers', ...
    'Inactive_layers','beta_star','Genre_consistency','Weak_consistency', ...
    'Weak_anchor_agreement','Embedding_silhouette','Agreement_with_proposed'});
numericVars = {'M','Total_layers','Active_layers','Inactive_layers','beta_star', ...
    'Genre_consistency','Weak_consistency','Weak_anchor_agreement', ...
    'Embedding_silhouette','Agreement_with_proposed'};
for v = 1:numel(numericVars)
    if iscell(summaryTbl.(numericVars{v}))
        summaryTbl.(numericVars{v}) = cell2mat(summaryTbl.(numericVars{v}));
    end
end

writetable(summaryTbl, fullfile(tableDir, 'split_advantage_algorithm_summary.csv'));
write_latex_table_simple(summaryTbl, fullfile(tableDir, 'split_advantage_algorithm_summary.tex'));
write_split_advantage_readme(summaryTbl, tableDir);
make_split_advantage_metric_figure(summaryTbl, figureDir);

fprintf('Done. Split-advantage outputs are in:\n');
fprintf('  %s\n', outDir);
end

function scenarios = split_advantage_design()
core8 = {'Action','Adventure','Comedy','Crime','Drama','Romance','Sci-Fi','Thriller'};
base = struct('tag', '', 'purpose', '', 'minUserRatings', 40, ...
    'maxNumUsers', 5000, 'selectedGenres', {core8}, ...
    'numStrongSplits', 4, 'numWeakSplits', 16, 'weakMovieFraction', 0.25, ...
    'weakEdgeKeepProb', 0.20, 'usePositiveOnly', false, 'positiveThreshold', 4.0);

scenarios = repmat(base, 1, 4);
scenarios(1) = set_stress_scenario(base, 'stress_u5000_anchor4_weak16_keep20', ...
    'Main stress test: 4 wide anchor layers and 16 weak layers per genre.', ...
    5000, 4, 16, 0.25, 0.20, false);
scenarios(2) = set_stress_scenario(base, 'stress_u5000_anchor4_weak24_keep10', ...
    'Harder stress test: more weak layers with stronger edge subsampling.', ...
    5000, 4, 24, 0.30, 0.10, false);
scenarios(3) = set_stress_scenario(base, 'stress_u8000_anchor4_weak16_keep20', ...
    'Same stress design with 8000 users.', ...
    8000, 4, 16, 0.25, 0.20, false);
scenarios(4) = set_stress_scenario(base, 'stress_positive4_u5000_anchor4_weak16_keep20', ...
    'Positive-only stress test using ratings >= 4.', ...
    5000, 4, 16, 0.25, 0.20, true);
end

function S = set_stress_scenario(base, tag, purpose, maxNumUsers, numStrongSplits, numWeakSplits, weakMovieFraction, weakEdgeKeepProb, usePositiveOnly)
S = base;
S.tag = tag;
S.purpose = purpose;
S.maxNumUsers = maxNumUsers;
S.numStrongSplits = numStrongSplits;
S.numWeakSplits = numWeakSplits;
S.weakMovieFraction = weakMovieFraction;
S.weakEdgeKeepProb = weakEdgeKeepProb;
S.usePositiveOnly = usePositiveOnly;
end

function [A_layers, layer_info, selectedUserIDs] = build_mixed_signal_layers(ratingsTbl, moviesTbl, S, rngSeed)
rng(rngSeed);

[userGroups, userIDs_unique] = findgroups(ratingsTbl.userId);
userCountVals = splitapply(@numel, ratingsTbl.movieId, userGroups);
keepMask = userCountVals >= S.minUserRatings;
activeUsers = userIDs_unique(keepMask);
activeCounts = userCountVals(keepMask);
[~,ord] = sort(activeCounts, 'descend');
activeUsers = activeUsers(ord);
activeUsers = activeUsers(1:min(S.maxNumUsers, numel(activeUsers)));
selectedUserIDs = sort(activeUsers);

ratingsTbl = ratingsTbl(ismember(ratingsTbl.userId, activeUsers), :);
if S.usePositiveOnly
    ratingsTbl = ratingsTbl(ratingsTbl.rating >= S.positiveThreshold, :);
end
if isempty(ratingsTbl)
    error('No ratings remain after filtering.');
end

n = numel(selectedUserIDs);
userRowMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
for i = 1:n
    userRowMap(selectedUserIDs(i)) = i;
end

movieGenreLists = cell(height(moviesTbl),1);
for i = 1:height(moviesTbl)
    g = moviesTbl.genres{i};
    if strcmp(g, '(no genres listed)')
        movieGenreLists{i} = {};
    else
        movieGenreLists{i} = strsplit(g, '|');
    end
end

A_layers = {};
layer_id = [];
genre = {};
split_id = [];
layer_type = {};
n_users = [];
n_movies = [];
n_edges = [];
beta_l = [];
density = [];
movie_ids = {};
edge_keep_prob = [];

layerCounter = 0;
for g = 1:numel(S.selectedGenres)
    thisGenre = S.selectedGenres{g};
    isGenreMovie = false(height(moviesTbl),1);
    for i = 1:height(moviesTbl)
        isGenreMovie(i) = any(strcmp(movieGenreLists{i}, thisGenre));
    end

    genreMovieIDs = moviesTbl.movieId(isGenreMovie);
    genreRatings = ratingsTbl(ismember(ratingsTbl.movieId, genreMovieIDs), :);
    if isempty(genreRatings)
        continue;
    end

    allMovies = unique(genreRatings.movieId);
    allMovies = allMovies(randperm(numel(allMovies)));
    minWeakMovies = S.numWeakSplits * 2;
    minStrongMovies = S.numStrongSplits * 2;
    maxWeakMovies = numel(allMovies) - minStrongMovies;
    if maxWeakMovies < minWeakMovies
        warning('Skipping genre %s: only %d movies after filtering, not enough for %d anchor and %d weak splits.', ...
            thisGenre, numel(allMovies), S.numStrongSplits, S.numWeakSplits);
        continue;
    end
    weakMovieCount = round(S.weakMovieFraction * numel(allMovies));
    weakMovieCount = min(max(weakMovieCount, minWeakMovies), maxWeakMovies);

    weakMovies = allMovies(1:weakMovieCount);
    strongMovies = allMovies(weakMovieCount+1:end);

    strongChunks = split_movie_ids(strongMovies, S.numStrongSplits);
    weakChunks = split_movie_ids(weakMovies, S.numWeakSplits);

    for j = 1:numel(strongChunks)
        [layerCounter, A_layers, layer_id, genre, split_id, layer_type, ...
            n_users, n_movies, n_edges, beta_l, density, movie_ids, edge_keep_prob] = ...
            append_layer(layerCounter, A_layers, layer_id, genre, split_id, layer_type, ...
            n_users, n_movies, n_edges, beta_l, density, movie_ids, edge_keep_prob, ...
            genreRatings, strongChunks{j}, userRowMap, n, thisGenre, j, 'anchor', 1.0);
    end

    for j = 1:numel(weakChunks)
        [layerCounter, A_layers, layer_id, genre, split_id, layer_type, ...
            n_users, n_movies, n_edges, beta_l, density, movie_ids, edge_keep_prob] = ...
            append_layer(layerCounter, A_layers, layer_id, genre, split_id, layer_type, ...
            n_users, n_movies, n_edges, beta_l, density, movie_ids, edge_keep_prob, ...
            genreRatings, weakChunks{j}, userRowMap, n, thisGenre, j, 'weak', S.weakEdgeKeepProb);
    end
end

layer_info = table(layer_id, genre, split_id, layer_type, n_users, n_movies, ...
    n_edges, beta_l, density, movie_ids, edge_keep_prob, ...
    'VariableNames', {'layer_id','genre','split_id','layer_type','n_users', ...
    'n_movies','n_edges','beta_l','density','movie_ids','edge_keep_prob'});
end

function chunks = split_movie_ids(movieIDs, numSplits)
N = numel(movieIDs);
edges = round(linspace(0, N, numSplits + 1));
chunks = cell(numSplits,1);
for s = 1:numSplits
    idx1 = edges(s) + 1;
    idx2 = edges(s+1);
    chunks{s} = movieIDs(idx1:idx2);
end
end

function [layerCounter, A_layers, layer_id, genre, split_id, layer_type, n_users, n_movies, n_edges, beta_l, density, movie_ids, edge_keep_prob] = append_layer(layerCounter, A_layers, layer_id, genre, split_id, layer_type, n_users, n_movies, n_edges, beta_l, density, movie_ids, edge_keep_prob, genreRatings, subMovies, userRowMap, n, thisGenre, splitId, typeName, keepProb)
subMovies = subMovies(:);
n_l = numel(subMovies);
if n_l < 2
    return;
end
layerRatings = genreRatings(ismember(genreRatings.movieId, subMovies), :);
if isempty(layerRatings)
    return;
end

movieColMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
for j = 1:n_l
    movieColMap(subMovies(j)) = j;
end

pairs = unique([layerRatings.userId, layerRatings.movieId], 'rows');
if keepProb < 1
    pairs = pairs(rand(size(pairs,1),1) <= keepProb, :);
end
if isempty(pairs)
    return;
end

rowIdx = zeros(size(pairs,1),1);
colIdx = zeros(size(pairs,1),1);
keep = false(size(pairs,1),1);
for k = 1:size(pairs,1)
    uid = pairs(k,1);
    mid = pairs(k,2);
    if isKey(userRowMap, uid) && isKey(movieColMap, mid)
        rowIdx(k) = userRowMap(uid);
        colIdx(k) = movieColMap(mid);
        keep(k) = true;
    end
end
rowIdx = rowIdx(keep);
colIdx = colIdx(keep);
if isempty(rowIdx)
    return;
end

A_l = sparse(rowIdx, colIdx, 1, n, n_l);
if nnz(A_l) == 0
    return;
end

layerCounter = layerCounter + 1;
A_layers{layerCounter,1} = A_l;
layer_id(layerCounter,1) = layerCounter;
genre{layerCounter,1} = thisGenre;
split_id(layerCounter,1) = splitId;
layer_type{layerCounter,1} = typeName;
n_users(layerCounter,1) = n;
n_movies(layerCounter,1) = n_l;
n_edges(layerCounter,1) = nnz(A_l);
beta_l(layerCounter,1) = log(n_l) / log(n);
density(layerCounter,1) = nnz(A_l) / (n * n_l);
movie_ids{layerCounter,1} = subMovies;
edge_keep_prob(layerCounter,1) = keepProb;
end

function tf = stress_scenario_needs_run_work(resultDir, tag, MList, overwrite)
if overwrite
    tf = true;
    return;
end
for M = MList
    if ~isfile(fullfile(resultDir, sprintf('%s_M%d.mat', tag, M)))
        tf = true;
        return;
    end
end
tf = false;
end

function R = run_stress_algorithms(A_layers, layer_info, Theta_full, Khat_all, M, rngSeed)
rng(rngSeed);
numAnchor = sum(strcmp(layer_info.layer_type, 'anchor'));
beta_star = choose_active_threshold_count(layer_info.beta_l, numAnchor);
active_idx = find(layer_info.beta_l >= beta_star);
inactive_idx = find(layer_info.beta_l < beta_star);

Theta_hat = Theta_full(active_idx, active_idx);
V_active = spectral_embed_from_similarity(Theta_hat, M);
cluster_labels = kmeans(V_active, M, 'Replicates', 40, 'MaxIter', 1000);

U_group = estimate_group_subspaces(A_layers, active_idx, cluster_labels, Khat_all, M);
inactive_results = classify_inactive_layers_genre(A_layers, layer_info, inactive_idx, U_group);
proposedLabels = get_movielens_final_labels(numel(A_layers), active_idx, cluster_labels, inactive_idx, inactive_results);

V_all_embed = spectral_embed_from_similarity(Theta_full, max(2, M));
allLayerLabels = kmeans(V_all_embed(:,1:M), M, 'Replicates', 40, 'MaxIter', 1000);
nearestLabels = classify_by_embedding_centroid(V_all_embed, active_idx, inactive_idx, cluster_labels, M);
sizeDensityLabels = size_density_kmeans(layer_info, M);

alg_labels = struct();
alg_labels.proposed_split_subspace = proposedLabels;
alg_labels.all_layer_spectral = allLayerLabels;
alg_labels.split_embedding_nearest = nearestLabels;
alg_labels.size_density_kmeans = sizeDensityLabels;

R = struct();
R.beta_star = beta_star;
R.M = M;
R.active_idx = active_idx;
R.inactive_idx = inactive_idx;
R.cluster_labels = cluster_labels;
R.inactive_results = inactive_results;
R.Theta_full = Theta_full;
R.Theta_hat = Theta_hat;
R.V_all_embed = V_all_embed;
R.U_group = U_group;
R.alg_labels = alg_labels;
R.layer_info = layer_info;
end

function beta_star = choose_active_threshold_count(betaVals, numActiveTarget)
betaVals = sort(betaVals(:), 'descend');
numActiveTarget = min(max(numActiveTarget, 2), numel(betaVals)-1);
beta_star = 0.5 * (betaVals(numActiveTarget) + betaVals(numActiveTarget + 1));
end

function U_group = estimate_group_subspaces(A_layers, active_idx, cluster_labels, Khat_all, M)
U_group = cell(M,1);
A_active = A_layers(active_idx);
for m = 1:M
    idx_m = find(cluster_labels == m);
    if isempty(idx_m)
        U_group{m} = [];
        continue;
    end
    K_group = max(1, round(median(Khat_all(active_idx(idx_m)))));
    A_concat = horzcat(A_active{idx_m});
    U_group{m} = left_svd_subspace(A_concat, K_group);
end
end

function labels = classify_by_embedding_centroid(V, active_idx, inactive_idx, active_labels, M)
labels = zeros(size(V,1),1);
labels(active_idx) = active_labels(:);
centroids = zeros(M, size(V,2));
for m = 1:M
    idx = active_idx(active_labels(:) == m);
    centroids(m,:) = mean(V(idx,:), 1);
end
for t = 1:numel(inactive_idx)
    ell = inactive_idx(t);
    d = sum((centroids - V(ell,:)).^2, 2);
    [~,labels(ell)] = min(d);
end
end

function labels = size_density_kmeans(layer_info, M)
X = [layer_info.beta_l, log1p(layer_info.n_movies), ...
     log1p(layer_info.n_edges), layer_info.density];
for j = 1:size(X,2)
    sigma = std(X(:,j));
    if sigma == 0 || ~isfinite(sigma)
        sigma = 1;
    end
    X(:,j) = (X(:,j) - mean(X(:,j))) / sigma;
end
labels = kmeans(X, M, 'Replicates', 40, 'MaxIter', 1000);
end

function metrics = stress_metrics(layer_info, labels, active_idx, inactive_idx, M, V)
metrics = struct();
metrics.genreConsistency = genre_consistency_score(layer_info, labels, M);
metrics.weakConsistency = weak_genre_consistency_score(layer_info, labels, inactive_idx, M);
metrics.weakAnchorAgreement = weak_anchor_agreement_score(layer_info, labels, active_idx, inactive_idx, M);
metrics.embeddingSilhouette = embedding_silhouette_score(V(:,1:2), labels);
end

function score = genre_consistency_score(layer_info, labels, M)
genres = unique(string(layer_info.genre), 'stable');
vals = zeros(numel(genres),1);
for g = 1:numel(genres)
    idx = string(layer_info.genre) == genres(g);
    vals(g) = max_cluster_fraction(labels(idx), M);
end
score = mean(vals);
end

function score = weak_genre_consistency_score(layer_info, labels, inactive_idx, M)
genres = unique(string(layer_info.genre), 'stable');
vals = NaN(numel(genres),1);
for g = 1:numel(genres)
    idx = inactive_idx(string(layer_info.genre(inactive_idx)) == genres(g));
    if ~isempty(idx)
        vals(g) = max_cluster_fraction(labels(idx), M);
    end
end
score = mean_omitnan(vals);
end

function score = weak_anchor_agreement_score(layer_info, labels, active_idx, inactive_idx, M)
genres = unique(string(layer_info.genre), 'stable');
vals = NaN(numel(genres),1);
for g = 1:numel(genres)
    activeG = active_idx(string(layer_info.genre(active_idx)) == genres(g));
    weakG = inactive_idx(string(layer_info.genre(inactive_idx)) == genres(g));
    if isempty(activeG) || isempty(weakG)
        continue;
    end
    counts = zeros(1,M);
    for m = 1:M
        counts(m) = sum(labels(activeG) == m);
    end
    [~,anchorCluster] = max(counts);
    vals(g) = mean(labels(weakG) == anchorCluster);
end
score = mean_omitnan(vals);
end

function frac = max_cluster_fraction(x, M)
counts = zeros(1,M);
for m = 1:M
    counts(m) = sum(x(:) == m);
end
frac = max(counts) / max(1, sum(counts));
end

function make_stress_algorithm_embedding_figure(layer_info, R, M, tag, figureDir)
algNames = fieldnames(R.alg_labels);
colors = lines(M);
fig = figure('Color','w', 'Position', [80 80 1180 920]);
tiledlayout(2,2, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:min(4,numel(algNames))
    nexttile;
    labels = R.alg_labels.(algNames{a});
    hold on;
    for m = 1:M
        idxA = intersect(R.active_idx(:), find(labels == m));
        idxI = intersect(R.inactive_idx(:), find(labels == m));
        scatter(R.V_all_embed(idxA,1), R.V_all_embed(idxA,2), 34, colors(m,:), 'filled');
        scatter(R.V_all_embed(idxI,1), R.V_all_embed(idxI,2), 42, ...
            'MarkerEdgeColor', colors(m,:), 'MarkerFaceColor', 'none', 'LineWidth', 1.0);
    end
    add_genre_labels(layer_info, R.V_all_embed);
    hold off;
    title(strrep(algNames{a}, '_', ' '), 'Interpreter', 'none');
    grid on; box on;
    hide_toolbar(gca);
end

sgtitle(sprintf('%s, M=%d, anchor layers filled / weak layers hollow', tag, M), 'Interpreter', 'none');
export_pair(fig, fullfile(figureDir, sprintf('stress_algorithm_embedding_%s_M%d', tag, M)));
close(fig);
end

function score = embedding_silhouette_score(X, labels)
labels = labels(:);
if numel(unique(labels)) < 2
    score = NaN;
    return;
end
n = size(X,1);
D = zeros(n,n);
for i = 1:n
    diffs = X - X(i,:);
    D(i,:) = sqrt(sum(diffs.^2, 2));
end
s = NaN(n,1);
allLabels = unique(labels);
for i = 1:n
    own = labels == labels(i);
    own(i) = false;
    if any(own)
        a = mean(D(i,own));
    else
        a = 0;
    end
    otherLabels = setdiff(allLabels, labels(i));
    bVals = zeros(numel(otherLabels),1);
    for k = 1:numel(otherLabels)
        bVals(k) = mean(D(i, labels == otherLabels(k)));
    end
    b = min(bVals);
    s(i) = (b - a) / max(a, b);
end
score = mean_omitnan(s);
end

function add_genre_labels(layer_info, V)
genres = unique(string(layer_info.genre), 'stable');
for g = 1:numel(genres)
    idx = find(string(layer_info.genre) == genres(g));
    xy = median(V(idx,1:2), 1);
    text(xy(1), xy(2), char(genres(g)), 'FontSize', 7, ...
        'HorizontalAlignment', 'center', 'BackgroundColor', 'w', ...
        'Margin', 1, 'Interpreter', 'none');
end
end

function make_split_advantage_metric_figure(T, figureDir)
scenarios = unique(string(T.Scenario), 'stable');
algs = unique(string(T.Algorithm), 'stable');
metrics = {'Weak_anchor_agreement','Weak_consistency','Genre_consistency'};
titles = {'Weak-to-anchor agreement', 'Weak-layer consistency', 'All-layer genre consistency'};

for M = [2 3]
    fig = figure('Color','w', 'Position', [80 80 1320 450]);
    tiledlayout(1,3, 'TileSpacing','compact', 'Padding','compact');
    for p = 1:numel(metrics)
        nexttile;
        hold on;
        for a = 1:numel(algs)
            vals = NaN(numel(scenarios),1);
            for s = 1:numel(scenarios)
                idx = T.M == M & string(T.Scenario) == scenarios(s) & string(T.Algorithm) == algs(a);
                vals(s) = T.(metrics{p})(idx);
            end
            plot(1:numel(scenarios), vals, '-o', 'LineWidth', 1.5, ...
                'DisplayName', strrep(char(algs(a)), '_', ' '));
        end
        hold off;
        xticks(1:numel(scenarios));
        xticklabels(scenarios);
        xtickangle(30);
        ylim([0 1.05]);
        title(titles{p});
        grid on; box on;
        legend('Location','best');
        hide_toolbar(gca);
    end
    sgtitle(sprintf('Split-advantage stress metrics, M=%d', M));
    export_pair(fig, fullfile(figureDir, sprintf('split_advantage_metrics_M%d', M)));
    close(fig);
end
end

function write_split_advantage_readme(T, tableDir)
outFile = fullfile(tableDir, 'split_advantage_readme.txt');
fid = fopen(outFile, 'w');
if fid < 0
    error('Could not write %s', outFile);
end
cleaner = onCleanup(@() fclose(fid));

fprintf(fid, 'MovieLens split-advantage stress tests\n');
fprintf(fid, '=====================================\n\n');
fprintf(fid, 'These scenarios create wide/high-signal anchor layers and many thin/edge-subsampled weak layers.\n');
fprintf(fid, 'The most relevant metric is Weak_anchor_agreement: weak layers from a genre should be assigned to the same cluster as that genre''s anchor layers.\n\n');
for M = [2 3]
    fprintf(fid, 'M = %d averages:\n', M);
    S = T(T.M == M,:);
    algs = unique(string(S.Algorithm), 'stable');
    for a = 1:numel(algs)
        A = S(string(S.Algorithm) == algs(a),:);
        fprintf(fid, '  %s: weak-anchor=%.3f, weak-consistency=%.3f, genre-consistency=%.3f, agreement-with-proposed=%.3f\n', ...
            char(algs(a)), mean_omitnan(A.Weak_anchor_agreement), ...
            mean_omitnan(A.Weak_consistency), mean_omitnan(A.Genre_consistency), ...
            mean_omitnan(A.Agreement_with_proposed));
    end
    fprintf(fid, '\n');
end
end

function agreement = label_agreement(labelsA, labelsB, M)
miss = layer_misclassification(labelsB, labelsA, M);
agreement = 1 - miss / numel(labelsA);
end

function miss = layer_misclassification(predLabels, refLabels, M)
predLabels = predLabels(:)';
refLabels = refLabels(:)';
permsM = perms(1:M);
missVals = zeros(size(permsM,1),1);
for p = 1:size(permsM,1)
    relabeled = permsM(p, refLabels);
    missVals(p) = sum(predLabels ~= relabeled);
end
miss = min(missVals);
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
    K = minDim - 1;
end
K = max(K, 1);
[U,~,~] = svds(A, K);
end

function x = mean_omitnan(v)
v = v(isfinite(v));
if isempty(v)
    x = NaN;
else
    x = mean(v);
end
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
