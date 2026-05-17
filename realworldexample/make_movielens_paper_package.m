function make_movielens_paper_package()
%MAKE_MOVIELENS_PAPER_PACKAGE Curate MovieLens outputs into a paper-facing package.
%
% Run after:
%   run_movielens_experiment_grid

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot)
    projectRoot = pwd;
end
addpath(projectRoot);

tableDir = fullfile(projectRoot, 'tables_movielens');
figureDir = fullfile(projectRoot, 'figures_movielens');
buildDir = fullfile(projectRoot, 'results_movielens_builds');
resultDir = fullfile(projectRoot, 'results_movielens');
paperDir = fullfile(projectRoot, 'paper_movielens');
paperFigDir = fullfile(paperDir, 'figures');
paperTableDir = fullfile(paperDir, 'tables');

ensure_dir(paperDir);
ensure_dir(paperFigDir);
ensure_dir(paperTableDir);

selectedRuns = { ...
    'Main M=3',             'uneven_all',       0.75, 3; ...
    'Coarse M=2',           'uneven_all',       0.75, 2; ...
    'High threshold M=3',   'uneven_all',       0.80, 3; ...
    'Over-split M=4',       'uneven_all',       0.75, 4; ...
    'Equal split robust.',  'equal_all',        0.75, 3; ...
    'Positive pref.',       'uneven_positive4', 0.80, 3};

copy_core_outputs(tableDir, figureDir, paperFigDir, paperTableDir);
make_selected_run_table(selectedRuns, tableDir, paperTableDir);
make_active_layer_catalog(buildDir, paperTableDir);
make_selected_cluster_summaries(selectedRuns, tableDir, paperTableDir);
make_story_guide(selectedRuns, tableDir, paperDir);

fprintf('Paper-facing MovieLens package written to:\n');
fprintf('  %s\n', paperDir);
fprintf('Suggested main figures are in:\n');
fprintf('  %s\n', paperFigDir);
fprintf('Suggested main tables are in:\n');
fprintf('  %s\n', paperTableDir);

% Keep the resultDir variable used by the README text below visible to
% MATLAB's dependency scanner for this experiment.
if ~exist(resultDir, 'dir')
    warning('Result directory not found: %s', resultDir);
end
end

function copy_core_outputs(tableDir, figureDir, paperFigDir, paperTableDir)
copy_if_exists(fullfile(tableDir, 'movielens_preprocessing_summary.csv'), paperTableDir);
copy_if_exists(fullfile(tableDir, 'movielens_preprocessing_summary.tex'), paperTableDir);
copy_if_exists(fullfile(tableDir, 'layer_summary_uneven_all.csv'), paperTableDir);
copy_if_exists(fullfile(tableDir, 'layer_summary_uneven_all.tex'), paperTableDir);
copy_if_exists(fullfile(tableDir, 'active_nonactive_counts_all_runs.csv'), paperTableDir);
copy_if_exists(fullfile(tableDir, 'active_nonactive_counts_all_runs.tex'), paperTableDir);
copy_if_exists(fullfile(tableDir, 'movielens_experiment_outcomes.txt'), paperTableDir);

figBases = { ...
    'beta_width_barplot_uneven_all'; ...
    'embedding_active_inactive_uneven_all_beta0.75_M3'; ...
    'cluster_composition_uneven_all_beta0.75_M3'; ...
    'embedding_active_inactive_uneven_all_beta0.75_M2'; ...
    'cluster_composition_uneven_all_beta0.75_M2'; ...
    'embedding_active_inactive_uneven_all_beta0.75_M4'; ...
    'cluster_composition_uneven_all_beta0.75_M4'; ...
    'cluster_composition_equal_all_beta0.75_M3'; ...
    'cluster_composition_uneven_positive4_beta0.80_M3'; ...
    'active_counts_sensitivity_uneven_all'};

for i = 1:numel(figBases)
    copy_if_exists(fullfile(figureDir, [figBases{i} '.png']), paperFigDir);
    copy_if_exists(fullfile(figureDir, [figBases{i} '.pdf']), paperFigDir);
end
end

function make_selected_run_table(selectedRuns, tableDir, paperTableDir)
countsFile = fullfile(tableDir, 'active_nonactive_counts_all_runs.csv');
clusterFile = fullfile(tableDir, 'cluster_summary_all_runs.csv');
if ~isfile(countsFile) || ~isfile(clusterFile)
    warning('Cannot make selected run table because summary CSVs are missing.');
    return;
end

countsTbl = readtable(countsFile);
clusterTbl = readtable(clusterFile);

Role = cell(size(selectedRuns,1),1);
Build = cell(size(selectedRuns,1),1);
beta_star = zeros(size(selectedRuns,1),1);
M = zeros(size(selectedRuns,1),1);
Active_layers = zeros(size(selectedRuns,1),1);
Non_active_layers = zeros(size(selectedRuns,1),1);
Total_layers = zeros(size(selectedRuns,1),1);
Interpretation = cell(size(selectedRuns,1),1);

for i = 1:size(selectedRuns,1)
    Role{i} = selectedRuns{i,1};
    Build{i} = selectedRuns{i,2};
    beta_star(i) = selectedRuns{i,3};
    M(i) = selectedRuns{i,4};

    mask = string(countsTbl.Build) == string(Build{i}) & ...
           abs(countsTbl.beta_star - beta_star(i)) < 1e-8 & ...
           countsTbl.M == M(i);
    if any(mask)
        c = countsTbl(find(mask, 1), :);
        Active_layers(i) = c.Active_layers;
        Non_active_layers(i) = c.Non_active_layers;
        Total_layers(i) = c.Total_layers;
    end

    Interpretation{i} = summarize_run_clusters(clusterTbl, Build{i}, beta_star(i), M(i));
end

T = table(Role, Build, beta_star, M, Active_layers, Non_active_layers, Total_layers, Interpretation);
writetable(T, fullfile(paperTableDir, 'selected_run_story_table.csv'));
write_latex_table_simple(T, fullfile(paperTableDir, 'selected_run_story_table.tex'));
end

function make_active_layer_catalog(buildDir, paperTableDir)
buildFile = fullfile(buildDir, 'movielens_build_users8000_splits8_uneven_all_seed7.mat');
if ~isfile(buildFile)
    warning('Cannot make active layer catalog because uneven_all build is missing.');
    return;
end

B = load(buildFile, 'layer_info');
layer_info = B.layer_info;
if ~ismember('density', layer_info.Properties.VariableNames)
    layer_info.density = layer_info.n_edges ./ (layer_info.n_users .* layer_info.n_movies);
end

T = layer_info(:, {'layer_id','genre','split_id','n_movies','n_edges','beta_l','density'});
T.active_beta070 = T.beta_l >= 0.70;
T.active_beta075 = T.beta_l >= 0.75;
T.active_beta080 = T.beta_l >= 0.80;
T.active_beta085 = T.beta_l >= 0.85;

writetable(T, fullfile(paperTableDir, 'uneven_all_layer_active_catalog.csv'));
write_latex_table_simple(T, fullfile(paperTableDir, 'uneven_all_layer_active_catalog.tex'));

G = groupsummary(T, 'genre', 'sum', {'active_beta070','active_beta075','active_beta080','active_beta085'});
G.Properties.VariableNames = strrep(G.Properties.VariableNames, 'sum_', '');
writetable(G, fullfile(paperTableDir, 'active_layers_by_genre_threshold.csv'));
write_latex_table_simple(G, fullfile(paperTableDir, 'active_layers_by_genre_threshold.tex'));
end

function make_selected_cluster_summaries(selectedRuns, tableDir, paperTableDir)
for i = 1:size(selectedRuns,1)
    buildtag = selectedRuns{i,2};
    beta_star = selectedRuns{i,3};
    M = selectedRuns{i,4};
    inBase = sprintf('cluster_composition_%s_beta%.2f_M%d', buildtag, beta_star, M);
    outBase = sprintf('%02d_%s', i, inBase);

    copy_if_exists(fullfile(tableDir, [inBase '.csv']), paperTableDir, [outBase '.csv']);
    copy_if_exists(fullfile(tableDir, [inBase '.tex']), paperTableDir, [outBase '.tex']);
end
end

function make_story_guide(selectedRuns, tableDir, paperDir)
outFile = fullfile(paperDir, 'README_paper_story.md');
fid = fopen(outFile, 'w');
if fid < 0
    error('Could not write %s', outFile);
end
cleaner = onCleanup(@() fclose(fid));

fprintf(fid, '# MovieLens Paper Story\n\n');
fprintf(fid, 'Use the full grid as supporting evidence, but lead with a focused story.\n\n');
fprintf(fid, '## Main Claim\n\n');
fprintf(fid, 'The active-layer threshold is not just a computational filter. In the uneven all-ratings construction, it keeps a balanced set of wide genre-split layers and lets the active-clustering plus inactive-classification pipeline recover interpretable between-layer structure.\n\n');
fprintf(fid, '## Recommended Main Sequence\n\n');
fprintf(fid, '1. `beta_width_barplot_uneven_all`: shows the active-layer selection mechanism.\n');
fprintf(fid, '2. `cluster_composition_uneven_all_beta0.75_M2`: coarse two-family structure.\n');
fprintf(fid, '3. `cluster_composition_uneven_all_beta0.75_M3`: main result with Action/Sci-Fi/Thriller, Comedy/Romance, and Crime/Drama/Romance-type families.\n');
fprintf(fid, '4. `embedding_active_inactive_uneven_all_beta0.75_M3`: shows inactive layers aligning with active clusters after classification.\n');
fprintf(fid, '5. `cluster_composition_uneven_all_beta0.75_M4`: diagnostic over-segmentation.\n');
fprintf(fid, '6. Equal-split and positive-only cluster composition plots as robustness/signal-change checks.\n\n');

fprintf(fid, '## Selected Runs\n\n');
for i = 1:size(selectedRuns,1)
    fprintf(fid, '- %s: `%s`, beta = %.2f, M = %d\n', ...
        selectedRuns{i,1}, selectedRuns{i,2}, selectedRuns{i,3}, selectedRuns{i,4});
end

fprintf(fid, '\n## Tables To Use\n\n');
fprintf(fid, '- `selected_run_story_table`: concise comparison of the six paper-facing runs.\n');
fprintf(fid, '- `active_layers_by_genre_threshold`: shows which genres contribute active layers as beta increases.\n');
fprintf(fid, '- `uneven_all_layer_active_catalog`: full layer-level audit trail for the main build.\n');
fprintf(fid, '- `movielens_preprocessing_summary` and `layer_summary_uneven_all`: preprocessing and construction context.\n');

if isfile(fullfile(tableDir, 'movielens_experiment_outcomes.txt'))
    fprintf(fid, '\nThe automatically generated detailed outcome summary is copied into `tables/movielens_experiment_outcomes.txt`.\n');
end
end

function s = summarize_run_clusters(clusterTbl, buildtag, beta_star, M)
mask = string(clusterTbl.Build) == string(buildtag) & ...
       abs(clusterTbl.beta_star - beta_star) < 1e-8 & ...
       clusterTbl.M == M;
C = clusterTbl(mask,:);
if height(C) == 0
    s = '';
    return;
end

parts = cell(height(C),1);
for r = 1:height(C)
    dom = table_text(C.Dominant_genres, r);
    parts{r} = sprintf('C%d: %s', C.Cluster(r), dom);
end
s = strjoin(parts, ' | ');
end

function s = table_text(col, idx)
if iscell(col)
    s = char(string(col{idx}));
else
    s = char(string(col(idx)));
end
end

function copy_if_exists(src, destDir, destName)
if nargin < 3
    [~,name,ext] = fileparts(src);
    destName = [name ext];
end
if isfile(src)
    copyfile(src, fullfile(destDir, destName));
end
end

function ensure_dir(d)
if ~exist(d, 'dir')
    mkdir(d);
end
end
