function make_movielens_realdata_tables(buildDir, resultDir, tableDir)
%MAKE_MOVIELENS_REALDATA_TABLES Create CSV and LaTeX MovieLens summaries.

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
if nargin < 3 || isempty(tableDir)
    tableDir = fullfile(projectRoot, 'tables_movielens');
end
if ~exist(tableDir, 'dir')
    mkdir(tableDir);
end

selectedGenres = {'Action','Adventure','Comedy','Crime','Drama','Romance','Sci-Fi','Thriller'};

buildFiles = dir(fullfile(buildDir, 'movielens_build_users8000_splits8_*_seed7.mat'));
runFiles = dir(fullfile(resultDir, 'movielens_*_users8000_splits8_beta*_M*_seed7.mat'));

write_preprocessing_summary(buildFiles, buildDir, tableDir);
write_layer_summaries(buildFiles, buildDir, tableDir);
[runIndex, clusterSummary] = write_run_tables(runFiles, resultDir, tableDir, selectedGenres);

if ~isempty(runIndex)
    writetable(runIndex, fullfile(tableDir, 'active_nonactive_counts_all_runs.csv'));
    write_latex_table_simple(runIndex, fullfile(tableDir, 'active_nonactive_counts_all_runs.tex'));
end

if ~isempty(clusterSummary)
    writetable(clusterSummary, fullfile(tableDir, 'cluster_summary_all_runs.csv'));
    write_latex_table_simple(clusterSummary, fullfile(tableDir, 'cluster_summary_all_runs.tex'));
end
end

function write_preprocessing_summary(buildFiles, buildDir, tableDir)
uniqueMovies = NaN;
if ~isempty(buildFiles)
    B = load(fullfile(buildDir, buildFiles(1).name), 'layer_info');
    if isfield(B, 'layer_info') && ismember('movie_ids', B.layer_info.Properties.VariableNames)
        allMovieIds = [];
        for i = 1:height(B.layer_info)
            allMovieIds = [allMovieIds; B.layer_info.movie_ids{i}(:)]; %#ok<AGROW>
        end
        uniqueMovies = numel(unique(allMovieIds));
    end
end

Metric = { ...
    'Original ratings'; ...
    'Original users'; ...
    'Original movies'; ...
    'Minimum ratings per retained user'; ...
    'Selected users'; ...
    'Genres used'; ...
    'Splits per genre'; ...
    'Candidate layers'};
Value = [25000095; 162541; 62423; 40; 8000; 8; 8; 64];

if ~isnan(uniqueMovies)
    Metric{end+1,1} = 'Unique movies used in constructed layers';
    Value(end+1,1) = uniqueMovies;
end

T = table(Metric, Value);
csvFile = fullfile(tableDir, 'movielens_preprocessing_summary.csv');
texFile = fullfile(tableDir, 'movielens_preprocessing_summary.tex');
writetable(T, csvFile);
write_latex_table_simple(T, texFile);
end

function write_layer_summaries(buildFiles, buildDir, tableDir)
for k = 1:numel(buildFiles)
    B = load(fullfile(buildDir, buildFiles(k).name));
    if ~isfield(B, 'layer_info')
        continue;
    end
    layer_info = ensure_density(B.layer_info);
    buildtag = infer_buildtag(B, buildFiles(k).name);

    Metric = {'n_l'; 'beta_l'; 'n_edges'; 'density'};
    vals = {layer_info.n_movies, layer_info.beta_l, layer_info.n_edges, layer_info.density};
    Min = zeros(numel(vals),1);
    Median = zeros(numel(vals),1);
    Max = zeros(numel(vals),1);
    for i = 1:numel(vals)
        x = vals{i};
        Min(i) = min(x);
        Median(i) = median(x);
        Max(i) = max(x);
    end
    T = table(Metric, Min, Median, Max);
    writetable(T, fullfile(tableDir, sprintf('layer_summary_%s.csv', buildtag)));
    write_latex_table_simple(T, fullfile(tableDir, sprintf('layer_summary_%s.tex', buildtag)));
end
end

function [runIndex, clusterSummary] = write_run_tables(runFiles, resultDir, tableDir, selectedGenres)
Build = cell(0,1);
beta_star_col = zeros(0,1);
M_col = zeros(0,1);
Active_layers = zeros(0,1);
Non_active_layers = zeros(0,1);
Total_layers = zeros(0,1);

summaryBuild = cell(0,1);
summaryBeta = zeros(0,1);
summaryM = zeros(0,1);
summaryCluster = zeros(0,1);
summaryNumLayers = zeros(0,1);
summaryActive = zeros(0,1);
summaryInactive = zeros(0,1);
summaryDominant = cell(0,1);

for k = 1:numel(runFiles)
    runPath = fullfile(resultDir, runFiles(k).name);
    R = load(runPath);
    if ~isfield(R, 'layer_info') || ~isfield(R, 'active_idx') || ~isfield(R, 'inactive_idx')
        continue;
    end

    buildtag = infer_buildtag(R, runFiles(k).name);
    beta_star = R.beta_star;
    M = R.M;
    numLayers = height(R.layer_info);
    final_labels = get_run_final_labels(R);

    Build{end+1,1} = buildtag; %#ok<AGROW>
    beta_star_col(end+1,1) = beta_star; %#ok<AGROW>
    M_col(end+1,1) = M; %#ok<AGROW>
    Active_layers(end+1,1) = numel(R.active_idx); %#ok<AGROW>
    Non_active_layers(end+1,1) = numel(R.inactive_idx); %#ok<AGROW>
    Total_layers(end+1,1) = numLayers; %#ok<AGROW>

    compTbl = movielens_cluster_composition(R.layer_info, final_labels, M, selectedGenres);
    betaText = sprintf('%.2f', beta_star);
    compBase = sprintf('cluster_composition_%s_beta%s_M%d', buildtag, betaText, M);
    writetable(compTbl, fullfile(tableDir, [compBase '.csv']));
    write_latex_table_simple(compTbl, fullfile(tableDir, [compBase '.tex']));

    layerGenres = string(R.layer_info.genre);
    for m = 1:M
        idx_m = final_labels(:) == m;
        summaryBuild{end+1,1} = buildtag; %#ok<AGROW>
        summaryBeta(end+1,1) = beta_star; %#ok<AGROW>
        summaryM(end+1,1) = M; %#ok<AGROW>
        summaryCluster(end+1,1) = m; %#ok<AGROW>
        summaryNumLayers(end+1,1) = sum(idx_m); %#ok<AGROW>
        summaryActive(end+1,1) = sum(idx_m & ismember((1:numLayers)', R.active_idx(:))); %#ok<AGROW>
        summaryInactive(end+1,1) = sum(idx_m & ismember((1:numLayers)', R.inactive_idx(:))); %#ok<AGROW>
        summaryDominant{end+1,1} = dominant_genre_string(layerGenres(idx_m), selectedGenres); %#ok<AGROW>
    end
end

runIndex = table(Build, beta_star_col, M_col, Active_layers, Non_active_layers, Total_layers, ...
    'VariableNames', {'Build','beta_star','M','Active_layers','Non_active_layers','Total_layers'});
if ~isempty(runIndex)
    runIndex = sortrows(runIndex, {'Build','M','beta_star'});
end

clusterSummary = table(summaryBuild, summaryBeta, summaryM, summaryCluster, ...
    summaryNumLayers, summaryActive, summaryInactive, summaryDominant, ...
    'VariableNames', {'Build','beta_star','M','Cluster','Number_of_layers', ...
                      'Active_layers','Non_active_layers','Dominant_genres'});
if ~isempty(clusterSummary)
    clusterSummary = sortrows(clusterSummary, {'Build','M','beta_star','Cluster'});
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

function s = dominant_genre_string(genresInCluster, selectedGenres)
parts = {};
counts = zeros(numel(selectedGenres),1);
for g = 1:numel(selectedGenres)
    counts(g) = sum(genresInCluster == string(selectedGenres{g}));
end
[countsSorted, ord] = sort(counts, 'descend');
for i = 1:numel(ord)
    if countsSorted(i) > 0
        parts{end+1} = sprintf('%s(%d)', selectedGenres{ord(i)}, countsSorted(i)); %#ok<AGROW>
    end
end
if isempty(parts)
    s = '';
else
    s = strjoin(parts, ', ');
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
