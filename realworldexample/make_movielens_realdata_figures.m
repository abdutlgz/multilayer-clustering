function make_movielens_realdata_figures(buildDir, resultDir, figureDir, makeHeatmaps)
%MAKE_MOVIELENS_REALDATA_FIGURES Create non-heatmap MovieLens figures.

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot)
    projectRoot = pwd;
end
addpath(projectRoot);

if nargin < 1 || isempty(buildDir)
    buildDir = fullfile(projectRoot, 'results_movielens_builds');
end
if nargin < 2 || isempty(resultDir)
    resultDir = fullfile(projectRoot, 'results_movielens');
end
if nargin < 3 || isempty(figureDir)
    figureDir = fullfile(projectRoot, 'figures_movielens');
end
if nargin < 4 || isempty(makeHeatmaps)
    makeHeatmaps = false;
end
if ~exist(figureDir, 'dir')
    mkdir(figureDir);
end

selectedGenres = {'Action','Adventure','Comedy','Crime','Drama','Romance','Sci-Fi','Thriller'};
buildFiles = dir(fullfile(buildDir, 'movielens_build_users8000_splits8_*_seed7.mat'));

for k = 1:numel(buildFiles)
    B = load(fullfile(buildDir, buildFiles(k).name));
    if ~isfield(B, 'layer_info')
        continue;
    end
    buildtag = infer_buildtag(B, buildFiles(k).name);
    layer_info = ensure_density(B.layer_info);
    make_beta_width_plot(layer_info, buildtag, selectedGenres, figureDir);
    make_active_count_sensitivity(buildtag, resultDir, figureDir);
end

selectedRuns = { ...
    'uneven_all', 0.75, 2; ...
    'uneven_all', 0.80, 3; ...
    'uneven_all', 0.75, 3; ...
    'equal_all', 0.75, 3; ...
    'uneven_positive4', 0.80, 3; ...
    'uneven_all', 0.75, 4};

for r = 1:size(selectedRuns,1)
    buildtag = selectedRuns{r,1};
    beta_star = selectedRuns{r,2};
    M = selectedRuns{r,3};
    runPath = find_run_file(resultDir, buildtag, beta_star, M);
    if isempty(runPath)
        warning('Missing selected run for figures: %s beta=%.2f M=%d', buildtag, beta_star, M);
        continue;
    end

    R = load(runPath);
    R = ensure_embedding_fields(R, runPath, M);
    make_embedding_plot(R, buildtag, beta_star, M, figureDir);
    make_cluster_composition_plot(R, buildtag, beta_star, M, selectedGenres, figureDir);
    if makeHeatmaps
        make_debug_heatmap(R, buildtag, beta_star, M, figureDir);
    end
end
end

function make_beta_width_plot(layer_info, buildtag, selectedGenres, figureDir)
[~,ord] = sort_layer_order(layer_info, selectedGenres);
info = layer_info(ord,:);

fig = figure('Color','w', 'Position', [100 100 1100 420]);
bar(1:height(info), info.beta_l, 0.85, 'FaceColor', [0.20 0.42 0.64], 'EdgeColor', 'none');
hold on;
thresholds = [0.70 0.75 0.80 0.85];
for i = 1:numel(thresholds)
    yline(thresholds(i), '--', sprintf('%.2f', thresholds(i)), ...
        'Color', [0.20 0.20 0.20], 'LineWidth', 1.0, 'LabelHorizontalAlignment', 'left');
end
genreSorted = string(info.genre);
breaks = find(genreSorted(1:end-1) ~= genreSorted(2:end));
for g = 1:numel(breaks)
    xline(breaks(g) + 0.5, ':', 'Color', [0.45 0.45 0.45]);
end
hold off;

ylim([0, max(1.0, max(info.beta_l) + 0.05)]);
xlim([0.25, height(info) + 0.75]);
xlabel('Layer index');
ylabel('\beta_l');
title(sprintf('MovieLens layer-width exponents (%s)', buildtag), 'Interpreter', 'none');
grid on;
box on;

ax = gca;
hide_toolbar(ax);
export_pair(fig, fullfile(figureDir, sprintf('beta_width_barplot_%s', buildtag)));
close(fig);
end

function make_active_count_sensitivity(buildtag, resultDir, figureDir)
fig = figure('Color','w', 'Position', [120 120 900 360]);
tiledlayout(1,2, 'TileSpacing', 'compact', 'Padding', 'compact');

for panel = 1:2
    M = panel + 1;
    nexttile;
    runFiles = dir(fullfile(resultDir, sprintf('movielens_%s_users8000_splits8_beta*_M%d_seed7.mat', buildtag, M)));
    betaVals = [];
    counts = [];
    for k = 1:numel(runFiles)
        R = load(fullfile(resultDir, runFiles(k).name), 'beta_star', 'active_idx', 'inactive_idx');
        betaVals(end+1,1) = R.beta_star; %#ok<AGROW>
        counts(end+1,:) = [numel(R.active_idx), numel(R.inactive_idx)]; %#ok<AGROW>
    end
    [betaVals,ord] = sort(betaVals);
    counts = counts(ord,:);
    if ~isempty(betaVals)
        bar(betaVals, counts, 'stacked');
        legend({'Active','Non-active'}, 'Location', 'best');
    end
    xlabel('\beta_*');
    ylabel('Number of layers');
    title(sprintf('M=%d', M));
    grid on;
    box on;
    hide_toolbar(gca);
end

sgtitle(sprintf('Active/non-active count sensitivity (%s)', buildtag), 'Interpreter', 'none');
export_pair(fig, fullfile(figureDir, sprintf('active_counts_sensitivity_%s', buildtag)));
close(fig);
end

function make_embedding_plot(R, buildtag, beta_star, M, figureDir)
final_labels = get_run_final_labels(R);
colors = lines(M);

fig = figure('Color','w', 'Position', [120 120 640 520]);
hold on;
for m = 1:M
    idxActive = intersect(R.active_idx(:), find(final_labels == m));
    idxInactive = intersect(R.inactive_idx(:), find(final_labels == m));

    scatter(R.V_all_embed(idxActive,1), R.V_all_embed(idxActive,2), ...
        70, colors(m,:), 'filled', 'o', 'DisplayName', sprintf('Cluster %d active', m));
    scatter(R.V_all_embed(idxInactive,1), R.V_all_embed(idxInactive,2), ...
        85, 'MarkerEdgeColor', colors(m,:), 'MarkerFaceColor', 'none', ...
        'Marker', 'o', 'LineWidth', 1.5, 'DisplayName', sprintf('Cluster %d non-active', m));
end
hold off;
xlabel('Embedding coordinate 1');
ylabel('Embedding coordinate 2');
title(sprintf('MovieLens active/non-active embedding (%s, beta=%.2f, M=%d)', buildtag, beta_star, M), ...
    'Interpreter', 'none');
grid on;
box on;
legend('Location', 'bestoutside');
hide_toolbar(gca);
export_pair(fig, fullfile(figureDir, sprintf('embedding_active_inactive_%s_beta%.2f_M%d', buildtag, beta_star, M)));
close(fig);
end

function make_cluster_composition_plot(R, buildtag, beta_star, M, selectedGenres, figureDir)
final_labels = get_run_final_labels(R);
compTbl = movielens_cluster_composition(R.layer_info, final_labels, M, selectedGenres);
counts = compTbl{:, 2:end};

fig = figure('Color','w', 'Position', [120 120 900 430]);
bar(counts, 'stacked');
xticks(1:numel(selectedGenres));
xticklabels(selectedGenres);
xtickangle(30);
ylabel('Number of layers');
xlabel('Genre');
title(sprintf('MovieLens cluster composition (%s, beta=%.2f, M=%d)', buildtag, beta_star, M), ...
    'Interpreter', 'none');
legend(arrayfun(@(m) sprintf('Cluster %d', m), 1:M, 'UniformOutput', false), ...
    'Location', 'bestoutside');
grid on;
box on;
hide_toolbar(gca);
export_pair(fig, fullfile(figureDir, sprintf('cluster_composition_%s_beta%.2f_M%d', buildtag, beta_star, M)));
close(fig);
end

function make_debug_heatmap(R, buildtag, beta_star, M, figureDir)
if ~isfield(R, 'Theta_full') || isempty(R.Theta_full)
    return;
end
final_labels = get_run_final_labels(R);
[~,ord] = sort(final_labels);

fig = figure('Color','w', 'Position', [120 120 560 500]);
imagesc(R.Theta_full(ord, ord));
axis square;
colorbar;
title(sprintf('Debug similarity heatmap (%s, beta=%.2f, M=%d)', buildtag, beta_star, M), ...
    'Interpreter', 'none');
hide_toolbar(gca);
export_pair(fig, fullfile(figureDir, sprintf('debug_heatmap_%s_beta%.2f_M%d', buildtag, beta_star, M)));
close(fig);
end

function R = ensure_embedding_fields(R, runPath, M)
needsSave = false;
if ~isfield(R, 'final_labels') || isempty(R.final_labels)
    R.final_labels = get_movielens_final_labels(height(R.layer_info), ...
        R.active_idx, R.cluster_labels, R.inactive_idx, R.inactive_results);
    final_labels = R.final_labels; %#ok<NASGU>
    save(runPath, 'final_labels', '-append');
end

if isfield(R, 'V_all_embed') && ~isempty(R.V_all_embed)
    return;
end

if isfield(R, 'Theta_full') && ~isempty(R.Theta_full)
    R.V_all_embed = spectral_embed_from_similarity(R.Theta_full, M);
    needsSave = true;
elseif isfield(R, 'A_layers') && ~isempty(R.A_layers)
    fprintf('Recomputing Theta_full and V_all_embed from A_layers for %s\n', runPath);
    [ThetaTmp, VTmp] = compute_movielens_theta_full(R.A_layers, 'M', M);
    R.Theta_full = ThetaTmp;
    R.V_all_embed = VTmp;
    needsSave = true;
else
    error('make_movielens_realdata_figures:MissingEmbeddingInputs', ...
        ['Cannot make embedding for %s: V_all_embed is missing, Theta_full is missing, ' ...
         'and A_layers is not available.'], runPath);
end

if needsSave
    Theta_full = R.Theta_full; %#ok<NASGU>
    V_all_embed = R.V_all_embed; %#ok<NASGU>
    save(runPath, 'Theta_full', 'V_all_embed', '-append');
end
end

function final_labels = get_run_final_labels(R)
if isfield(R, 'final_labels') && ~isempty(R.final_labels)
    final_labels = R.final_labels;
else
    final_labels = get_movielens_final_labels(height(R.layer_info), ...
        R.active_idx, R.cluster_labels, R.inactive_idx, R.inactive_results);
end
end

function runPath = find_run_file(resultDir, buildtag, beta_star, M)
pattern = sprintf('movielens_%s_users8000_splits8_beta%.2f_M%d_seed7.mat', buildtag, beta_star, M);
D = dir(fullfile(resultDir, pattern));
if isempty(D)
    runPath = '';
else
    runPath = fullfile(resultDir, D(1).name);
end
end

function [genreOrd, ord] = sort_layer_order(layer_info, selectedGenres)
genreOrd = zeros(height(layer_info),1);
layerGenre = string(layer_info.genre);
for g = 1:numel(selectedGenres)
    genreOrd(layerGenre == string(selectedGenres{g})) = g;
end
[~,ord] = sortrows([genreOrd, layer_info.split_id]);
end

function V_embed = spectral_embed_from_similarity(Theta, M)
Theta = (Theta + Theta') / 2;
n = size(Theta,1);
if M >= n
    [V,D] = eig(full(Theta));
    [~,ord] = sort(diag(D), 'descend');
    V_embed = V(:, ord(1:min(M,n)));
else
    try
        [V_embed,~] = eigs(Theta, M, 'largestreal');
    catch
        [V,D] = eig(full(Theta));
        [~,ord] = sort(diag(D), 'descend');
        V_embed = V(:, ord(1:M));
    end
end
rowNorms = sqrt(sum(V_embed.^2,2));
rowNorms(rowNorms == 0) = 1;
V_embed = V_embed ./ rowNorms;
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

function buildtag = infer_buildtag(S, fileName)
if isfield(S, 'buildtag') && ~isempty(S.buildtag)
    buildtag = char(string(S.buildtag));
    return;
end
if contains(fileName, 'uneven_positive4')
    buildtag = 'uneven_positive4';
elseif contains(fileName, 'equal_all') || contains(fileName, '_equal_')
    buildtag = 'equal_all';
else
    buildtag = 'uneven_all';
end
end

function layer_info = ensure_density(layer_info)
if ~ismember('density', layer_info.Properties.VariableNames)
    layer_info.density = layer_info.n_edges ./ (layer_info.n_users .* layer_info.n_movies);
end
end
