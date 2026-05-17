function run_movielens_scenario_algorithm_comparison()
%RUN_MOVIELENS_SCENARIO_ALGORITHM_COMPARISON Compare algorithms on scenarios.
%
% Run after:
%   run_movielens_realdata_scenarios
%
% Algorithms:
%   1. proposed_split_subspace: saved active-layer clustering plus subspace
%      classification of inactive layers.
%   2. all_layer_spectral: spectral clustering on all layers using Theta_full.
%   3. split_embedding_nearest: same active labels as proposed, but inactive
%      layers classified by nearest active centroid in the all-layer embedding.
%   4. size_density_kmeans: k-means using only beta_l, n_movies, n_edges,
%      and density. This is a negative-control baseline for "just width".

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot)
    projectRoot = pwd;
end
addpath(projectRoot);

scenarioDir = fullfile(projectRoot, 'movielens_scenario_experiments');
resultDir = fullfile(scenarioDir, 'results');
outDir = fullfile(scenarioDir, 'algorithm_comparison');
tableDir = fullfile(outDir, 'tables');
figureDir = fullfile(outDir, 'figures');
ensure_dir(outDir);
ensure_dir(tableDir);
ensure_dir(figureDir);

runFiles = dir(fullfile(resultDir, '*_M*.mat'));
summaryRows = cell(0, 10);

for f = 1:numel(runFiles)
    runPath = fullfile(resultDir, runFiles(f).name);
    R = load(runPath);
    if ~isfield(R, 'Theta_full') || ~isfield(R, 'layer_info') || ~isfield(R, 'M')
        continue;
    end

    scenarioTag = R.scenario.tag;
    M = R.M;
    layer_info = ensure_density(R.layer_info);
    selectedGenres = R.scenario.selectedGenres;

    algs = compute_algorithm_labels(R);
    algNames = fieldnames(algs);
    for a = 1:numel(algNames)
        algName = algNames{a};
        labels = algs.(algName);

        compTbl = movielens_cluster_composition(layer_info, labels, M, selectedGenres);
        compBase = fullfile(tableDir, sprintf('composition_%s_M%d_%s', scenarioTag, M, algName));
        writetable(compTbl, [compBase '.csv']);
        write_latex_table_simple(compTbl, [compBase '.tex']);

        genreConsistency = genre_consistency_score(layer_info, labels, M);
        sil = embedding_silhouette_score(R.V_all_embed(:,1:2), labels);
        ari = adjusted_rand_index(string(layer_info.genre), labels);
        agreement = label_agreement(R.final_labels, labels, M);

        summaryRows(end+1,:) = {scenarioTag, M, algName, height(layer_info), ...
            numel(R.active_idx), numel(R.inactive_idx), genreConsistency, ...
            sil, ari, agreement}; %#ok<AGROW>
    end

    if strcmp(scenarioTag, 'main_u8000_core8_s8')
        make_algorithm_embedding_figure(R, algs, figureDir);
    end
end

summaryTbl = cell2table(summaryRows, 'VariableNames', { ...
    'Scenario','M','Algorithm','Total_layers','Active_layers','Inactive_layers', ...
    'Genre_consistency','Embedding_silhouette','Genre_ARI','Agreement_with_proposed'});
numericVars = {'M','Total_layers','Active_layers','Inactive_layers', ...
    'Genre_consistency','Embedding_silhouette','Genre_ARI','Agreement_with_proposed'};
for v = 1:numel(numericVars)
    if iscell(summaryTbl.(numericVars{v}))
        summaryTbl.(numericVars{v}) = cell2mat(summaryTbl.(numericVars{v}));
    end
end

writetable(summaryTbl, fullfile(tableDir, 'algorithm_comparison_summary.csv'));
write_latex_table_simple(summaryTbl, fullfile(tableDir, 'algorithm_comparison_summary.tex'));
write_algorithm_comparison_guide(summaryTbl, tableDir);

fprintf('Algorithm comparison outputs are in:\n');
fprintf('  %s\n', outDir);
end

function algs = compute_algorithm_labels(R)
M = R.M;
algs = struct();
algs.proposed_split_subspace = R.final_labels(:);

V = spectral_embed_from_similarity(R.Theta_full, max(2, M));
algs.all_layer_spectral = kmeans(V(:,1:M), M, 'Replicates', 40, 'MaxIter', 1000);

labelsNearest = zeros(size(R.final_labels(:)));
labelsNearest(R.active_idx) = R.cluster_labels(:);
centroids = zeros(M, size(V,2));
for m = 1:M
    idx = R.active_idx(R.cluster_labels(:) == m);
    if isempty(idx)
        centroids(m,:) = NaN;
    else
        centroids(m,:) = mean(V(idx,:), 1);
    end
end
for t = 1:numel(R.inactive_idx)
    ell = R.inactive_idx(t);
    d = sum((centroids - V(ell,:)).^2, 2);
    [~,labelsNearest(ell)] = min(d);
end
algs.split_embedding_nearest = labelsNearest;

layer_info = ensure_density(R.layer_info);
X = [layer_info.beta_l, log1p(layer_info.n_movies), ...
     log1p(layer_info.n_edges), layer_info.density];
X = standardize_columns(X);
algs.size_density_kmeans = kmeans(X, M, 'Replicates', 40, 'MaxIter', 1000);
end

function make_algorithm_embedding_figure(R, algs, figureDir)
algNames = fieldnames(algs);
M = R.M;
fig = figure('Color','w', 'Position', [80 80 1180 920]);
tiledlayout(2,2, 'TileSpacing', 'compact', 'Padding', 'compact');
colors = lines(M);

for a = 1:min(4,numel(algNames))
    nexttile;
    labels = algs.(algNames{a});
    hold on;
    for m = 1:M
        idxA = intersect(R.active_idx(:), find(labels == m));
        idxI = intersect(R.inactive_idx(:), find(labels == m));
        scatter(R.V_all_embed(idxA,1), R.V_all_embed(idxA,2), 36, colors(m,:), 'filled');
        scatter(R.V_all_embed(idxI,1), R.V_all_embed(idxI,2), 44, ...
            'MarkerEdgeColor', colors(m,:), 'MarkerFaceColor', 'none', 'LineWidth', 1.0);
    end
    hold off;
    title(strrep(algNames{a}, '_', ' '), 'Interpreter', 'none');
    xlabel('Embedding coordinate 1');
    ylabel('Embedding coordinate 2');
    grid on; box on;
    hide_toolbar(gca);
end

sgtitle(sprintf('Main MovieLens scenario algorithm comparison, M=%d', M));
export_pair(fig, fullfile(figureDir, sprintf('main_algorithm_comparison_M%d', M)));
close(fig);
end

function score = genre_consistency_score(layer_info, labels, M)
genres = unique(string(layer_info.genre), 'stable');
vals = zeros(numel(genres),1);
for g = 1:numel(genres)
    idx = string(layer_info.genre) == genres(g);
    counts = zeros(1,M);
    for m = 1:M
        counts(m) = sum(idx & labels(:) == m);
    end
    vals(g) = max(counts) / sum(idx);
end
score = mean(vals);
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

function ari = adjusted_rand_index(groupA, groupB)
groupA = string(groupA(:));
groupB = groupB(:);
[~,~,ia] = unique(groupA);
[~,~,ib] = unique(groupB);
na = max(ia);
nb = max(ib);
N = zeros(na, nb);
for i = 1:numel(ia)
    N(ia(i), ib(i)) = N(ia(i), ib(i)) + 1;
end
comb2 = @(x) x .* (x - 1) / 2;
sumN = sum(comb2(N(:)));
sumRows = sum(comb2(sum(N,2)));
sumCols = sum(comb2(sum(N,1)));
total = comb2(numel(ia));
expected = sumRows * sumCols / total;
maxIndex = 0.5 * (sumRows + sumCols);
denom = maxIndex - expected;
if denom == 0
    ari = NaN;
else
    ari = (sumN - expected) / denom;
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

function X = standardize_columns(X)
for j = 1:size(X,2)
    mu = mean(X(:,j));
    sigma = std(X(:,j));
    if sigma == 0 || ~isfinite(sigma)
        sigma = 1;
    end
    X(:,j) = (X(:,j) - mu) / sigma;
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

function write_algorithm_comparison_guide(T, tableDir)
outFile = fullfile(tableDir, 'algorithm_comparison_readme.txt');
fid = fopen(outFile, 'w');
if fid < 0
    error('Could not write %s', outFile);
end
cleaner = onCleanup(@() fclose(fid));

fprintf(fid, 'MovieLens scenario algorithm comparison\n');
fprintf(fid, '======================================\n\n');
fprintf(fid, 'Use this as a diagnostic, not as a supervised benchmark. The metrics are interpretability and stability proxies because MovieLens genre families are not ground-truth layer clusters.\n\n');
for M = [2 3]
    fprintf(fid, 'M = %d, average over scenarios:\n', M);
    S = T(T.M == M,:);
    algs = unique(string(S.Algorithm), 'stable');
    for a = 1:numel(algs)
        A = S(string(S.Algorithm) == algs(a),:);
        fprintf(fid, '  %s: genre consistency=%.3f, silhouette=%.3f, genre ARI=%.3f, agreement with proposed=%.3f\n', ...
            char(algs(a)), mean_omitnan(A.Genre_consistency), ...
            mean_omitnan(A.Embedding_silhouette), mean_omitnan(A.Genre_ARI), ...
            mean_omitnan(A.Agreement_with_proposed));
    end
    fprintf(fid, '\n');
end
fprintf(fid, 'Suggested figures:\n');
fprintf(fid, '  figures/main_algorithm_comparison_M2.pdf\n');
fprintf(fid, '  figures/main_algorithm_comparison_M3.pdf\n');
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
