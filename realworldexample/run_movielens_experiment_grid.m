function run_movielens_experiment_grid(overwrite, makeHeatmaps)
%RUN_MOVIELENS_EXPERIMENT_GRID Build and run the MovieLens real-data grid.
%
% Run from the project root with:
%   run_movielens_experiment_grid

if nargin < 1 || isempty(overwrite)
    overwrite = false;
end
if nargin < 2 || isempty(makeHeatmaps)
    makeHeatmaps = false;
end

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot)
    projectRoot = pwd;
end
addpath(projectRoot);
addpath(fullfile(projectRoot, 'ml-25m'));

dataFolder = fullfile(projectRoot, 'ml-25m');
buildDir = fullfile(projectRoot, 'results_movielens_builds');
resultDir = fullfile(projectRoot, 'results_movielens');
tableDir = fullfile(projectRoot, 'tables_movielens');
figureDir = fullfile(projectRoot, 'figures_movielens');

ensure_dir(buildDir);
ensure_dir(resultDir);
ensure_dir(tableDir);
ensure_dir(figureDir);

minUserRatings = 40;
maxNumUsers = 8000;
selectedGenres = {'Action','Adventure','Comedy','Crime','Drama','Romance','Sci-Fi','Thriller'};
numSplits = 8;
rngSeed = 7;

buildSettings = struct( ...
    'buildtag', {'uneven_all', 'equal_all', 'uneven_positive4'}, ...
    'splitMode', {'uneven', 'equal', 'uneven'}, ...
    'usePositiveOnly', {false, false, true}, ...
    'positiveThreshold', {NaN, NaN, 4.0}, ...
    'buildFile', {'movielens_build_users8000_splits8_uneven_all_seed7.mat', ...
                  'movielens_build_users8000_splits8_equal_all_seed7.mat', ...
                  'movielens_build_users8000_splits8_uneven_positive4_seed7.mat'});

grid = [ ...
    2 0.70; 2 0.75; 2 0.80; ...
    3 0.70; 3 0.75; 3 0.80; 3 0.85; ...
    4 0.75; 4 0.80];

pipelineOpts = struct();
pipelineOpts.rank_method = 'energy';
pipelineOpts.energy_thresh = 0.90;
pipelineOpts.fixed_rank = 5;
pipelineOpts.maxRankTry = 10;
pipelineOpts.use_normalized_similarity = true;

fprintf('MovieLens experiment grid\n');
fprintf('  overwrite = %d\n', overwrite);
fprintf('  makeHeatmaps = %d\n', makeHeatmaps);

needBuild = overwrite;
for b = 1:numel(buildSettings)
    needBuild = needBuild || ~isfile(fullfile(buildDir, buildSettings(b).buildFile));
end

ratingsTbl = [];
moviesTbl = [];
if needBuild
    fprintf('Loading MovieLens 25M tables from %s ...\n', dataFolder);
    [ratingsTbl, moviesTbl] = load_movielens25_tables(dataFolder);
end

for b = 1:numel(buildSettings)
    S = buildSettings(b);
    buildFile = fullfile(buildDir, S.buildFile);

    if isfile(buildFile) && ~overwrite
        fprintf('Loading existing build: %s\n', buildFile);
        B = load(buildFile);
        A_layers = B.A_layers;
        layer_info = B.layer_info;
        user_ids = get_loaded_field(B, 'user_ids', []);
        layer_info = ensure_layer_info_density(layer_info);
        if ~isfield(B, 'buildtag') || ~ismember('density', B.layer_info.Properties.VariableNames)
            buildtag = S.buildtag; %#ok<NASGU>
            save(buildFile, 'layer_info', 'buildtag', '-append');
        end
    else
        if isempty(ratingsTbl) || isempty(moviesTbl)
            fprintf('Loading MovieLens 25M tables from %s ...\n', dataFolder);
            [ratingsTbl, moviesTbl] = load_movielens25_tables(dataFolder);
        end

        fprintf('Building %s layers...\n', S.buildtag);
        positiveThresholdForBuild = S.positiveThreshold;
        if isnan(positiveThresholdForBuild)
            positiveThresholdForBuild = 4.0;
        end

        [A_layers, layer_info, user_ids] = build_movielens_genre_splits( ...
            ratingsTbl, moviesTbl, ...
            minUserRatings, maxNumUsers, ...
            selectedGenres, numSplits, S.splitMode, rngSeed, ...
            S.usePositiveOnly, positiveThresholdForBuild);

        layer_info = ensure_layer_info_density(layer_info);
        splitMode = S.splitMode; %#ok<NASGU>
        usePositiveOnly = S.usePositiveOnly; %#ok<NASGU>
        positiveThreshold = S.positiveThreshold; %#ok<NASGU>
        buildtag = S.buildtag; %#ok<NASGU>

        fprintf('Saving build: %s\n', buildFile);
        save(buildFile, ...
            'A_layers', 'layer_info', 'user_ids', ...
            'maxNumUsers', 'minUserRatings', 'selectedGenres', 'numSplits', ...
            'splitMode', 'rngSeed', 'usePositiveOnly', 'positiveThreshold', ...
            'buildtag', '-v7.3');
    end

    if isempty(user_ids)
        user_ids = (1:size(A_layers{1},1))';
    end

    if ~build_needs_run_work(resultDir, S.buildtag, maxNumUsers, numSplits, rngSeed, grid, overwrite)
        fprintf('All complete runs already exist for build %s; skipping pipeline recomputation.\n', S.buildtag);
        continue;
    end

    fprintf('Computing full layer similarity once for build %s ...\n', S.buildtag);
    [Theta_full_base, ~, U_all, Khat_all] = compute_movielens_theta_full( ...
        A_layers, 'M', [], ...
        'rank_method', pipelineOpts.rank_method, ...
        'energy_thresh', pipelineOpts.energy_thresh, ...
        'fixed_rank', pipelineOpts.fixed_rank, ...
        'maxRankTry', pipelineOpts.maxRankTry, ...
        'use_normalized_similarity', pipelineOpts.use_normalized_similarity);

    for g = 1:size(grid,1)
        M = grid(g,1);
        beta_star = grid(g,2);
        runFile = fullfile(resultDir, sprintf( ...
            'movielens_%s_users%d_splits%d_beta%.2f_M%d_seed%d.mat', ...
            S.buildtag, maxNumUsers, numSplits, beta_star, M, rngSeed));

        if isfile(runFile) && ~overwrite
            fprintf('Skipping existing run: %s\n', runFile);
            ensure_existing_run_file_fields(runFile, A_layers, layer_info, user_ids, ...
                S.buildtag, rngSeed, Theta_full_base, M);
            continue;
        end

        fprintf('Running build=%s beta*=%.2f M=%d ...\n', S.buildtag, beta_star, M);
        try
            R = run_one_movielens_setting(A_layers, layer_info, beta_star, M, rngSeed, ...
                pipelineOpts, Theta_full_base, U_all, Khat_all);
            run_status = 'ok'; %#ok<NASGU>
            error_message = ''; %#ok<NASGU>
        catch ME
            warning('Run failed for build=%s beta*=%.2f M=%d: %s', ...
                S.buildtag, beta_star, M, ME.message);
            R = failed_movielens_result(A_layers, layer_info, beta_star, M, Theta_full_base);
            run_status = 'failed'; %#ok<NASGU>
            error_message = ME.message; %#ok<NASGU>
        end

        buildtag = S.buildtag; %#ok<NASGU>
        active_idx = R.active_idx; %#ok<NASGU>
        inactive_idx = R.inactive_idx; %#ok<NASGU>
        active_results = R.active_results; %#ok<NASGU>
        inactive_results = R.inactive_results; %#ok<NASGU>
        final_labels = R.final_labels; %#ok<NASGU>
        Theta_hat = R.Theta_hat; %#ok<NASGU>
        cluster_labels = R.cluster_labels; %#ok<NASGU>
        U_group = R.U_group; %#ok<NASGU>
        Khat_group = R.Khat_group; %#ok<NASGU>
        Theta_full = R.Theta_full; %#ok<NASGU>
        V_all_embed = R.V_all_embed; %#ok<NASGU>
        Khat_all_run = Khat_all; %#ok<NASGU>
        rank_method = pipelineOpts.rank_method; %#ok<NASGU>
        energy_thresh = pipelineOpts.energy_thresh; %#ok<NASGU>
        fixed_rank = pipelineOpts.fixed_rank; %#ok<NASGU>
        maxRankTry = pipelineOpts.maxRankTry; %#ok<NASGU>
        use_normalized_similarity = pipelineOpts.use_normalized_similarity; %#ok<NASGU>

        save(runFile, ...
            'beta_star', 'M', 'buildtag', 'rngSeed', ...
            'active_idx', 'inactive_idx', ...
            'active_results', 'inactive_results', 'final_labels', ...
            'Theta_hat', 'cluster_labels', 'U_group', 'Khat_group', ...
            'layer_info', 'user_ids', 'A_layers', ...
            'Theta_full', 'V_all_embed', 'Khat_all_run', ...
            'rank_method', 'energy_thresh', 'fixed_rank', 'maxRankTry', ...
            'use_normalized_similarity', 'run_status', 'error_message', '-v7.3');
    end
end

fprintf('Generating MovieLens tables...\n');
make_movielens_realdata_tables(buildDir, resultDir, tableDir);

fprintf('Generating MovieLens figures...\n');
make_movielens_realdata_figures(buildDir, resultDir, figureDir, makeHeatmaps);

fprintf('Writing outcome summary...\n');
write_movielens_outcome_summary(tableDir, resultDir);

fprintf('Curating paper-facing MovieLens package...\n');
make_movielens_paper_package();

fprintf('Done. Outputs are in:\n');
fprintf('  %s\n', buildDir);
fprintf('  %s\n', resultDir);
fprintf('  %s\n', tableDir);
fprintf('  %s\n', figureDir);
fprintf('  %s\n', fullfile(projectRoot, 'paper_movielens'));
end

function R = run_one_movielens_setting(A_layers, layer_info, beta_star, M, rngSeed, opts, Theta_full, U_all, Khat_all)
rng(rngSeed);

active_idx = find(layer_info.beta_l >= beta_star);
inactive_idx = find(layer_info.beta_l < beta_star);
numActive = numel(active_idx);

if numActive < M
    error('Only %d active layers for beta_star=%.2f, but M=%d.', numActive, beta_star, M);
end

Theta_hat = Theta_full(active_idx, active_idx);
V_embed = spectral_embed_from_similarity(Theta_hat, M);
cluster_labels = kmeans(V_embed, M, 'Replicates', 30, 'MaxIter', 1000);

U_group = cell(M,1);
Khat_group = zeros(M,1);
A_active = A_layers(active_idx);

for m = 1:M
    idx_m = find(cluster_labels == m);
    if isempty(idx_m)
        warning('Cluster %d is empty for beta_star=%.2f, M=%d.', m, beta_star, M);
        U_group{m} = [];
        Khat_group(m) = 0;
        continue;
    end

    K_group = max(1, round(median(Khat_all(active_idx(idx_m)))));
    Khat_group(m) = K_group;
    A_concat = horzcat(A_active{idx_m});
    U_group{m} = left_svd_subspace(A_concat, K_group, opts.maxRankTry);
end

if ~isempty(inactive_idx)
    inactive_results = classify_inactive_layers_genre(A_layers, layer_info, inactive_idx, U_group);
else
    inactive_results = table();
end

active_results = layer_info(active_idx, :);
active_results.original_layer_id = active_idx(:);
active_results.Khat = Khat_all(active_idx);
active_results.cluster = cluster_labels(:);

final_labels = get_movielens_final_labels(numel(A_layers), active_idx, cluster_labels, inactive_idx, inactive_results);
V_all_embed = spectral_embed_from_similarity(Theta_full, M);

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
R.Theta_full = Theta_full;
R.V_all_embed = V_all_embed;
end

function R = failed_movielens_result(A_layers, layer_info, beta_star, M, Theta_full)
active_idx = find(layer_info.beta_l >= beta_star);
inactive_idx = find(layer_info.beta_l < beta_star);
numLayers = numel(A_layers);

R = struct();
R.active_idx = active_idx;
R.inactive_idx = inactive_idx;
R.active_results = table();
R.inactive_results = table();
R.final_labels = zeros(numLayers, 1);
R.Theta_hat = Theta_full(active_idx, active_idx);
R.cluster_labels = zeros(numel(active_idx), 1);
R.U_group = cell(M,1);
R.Khat_group = zeros(M,1);
R.Theta_full = Theta_full;
R.V_all_embed = spectral_embed_from_similarity(Theta_full, min(M, max(1, numLayers-1)));
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
rowNorms = sqrt(sum(V_embed.^2, 2));
rowNorms(rowNorms == 0) = 1;
V_embed = V_embed ./ rowNorms;
end

function U = left_svd_subspace(A, K, maxRankTry)
minDim = min(size(A));
K = min(K, minDim);
K = max(K, 1);
if K >= minDim
    if minDim <= 500
        [Ufull,~,~] = svd(full(A), 'econ');
        U = Ufull(:,1:K);
        return;
    end
    K = minDim - 1;
end
try
    [U,~,~] = svds(A, K);
catch
    rTry = min([maxRankTry, size(A,1)-1, size(A,2)-1]);
    rTry = max(rTry, K);
    [U0,~,~] = svds(A, rTry);
    U = U0(:,1:K);
end
end

function layer_info = ensure_layer_info_density(layer_info)
if ~ismember('density', layer_info.Properties.VariableNames)
    layer_info.density = layer_info.n_edges ./ (layer_info.n_users .* layer_info.n_movies);
end
end

function val = get_loaded_field(S, name, defaultVal)
if isfield(S, name)
    val = S.(name);
else
    val = defaultVal;
end
end

function ensure_dir(d)
if ~exist(d, 'dir')
    mkdir(d);
end
end

function tf = build_needs_run_work(resultDir, buildtag, maxNumUsers, numSplits, rngSeed, grid, overwrite)
if overwrite
    tf = true;
    return;
end
required = {'beta_star','M','buildtag','rngSeed','active_idx','inactive_idx', ...
    'active_results','inactive_results','final_labels','Theta_hat','cluster_labels', ...
    'U_group','Khat_group','layer_info','user_ids','A_layers','Theta_full','V_all_embed'};
for g = 1:size(grid,1)
    M = grid(g,1);
    beta_star = grid(g,2);
    runFile = fullfile(resultDir, sprintf( ...
        'movielens_%s_users%d_splits%d_beta%.2f_M%d_seed%d.mat', ...
        buildtag, maxNumUsers, numSplits, beta_star, M, rngSeed));
    if ~isfile(runFile)
        tf = true;
        return;
    end
    vars = whos('-file', runFile);
    names = {vars.name};
    if any(~ismember(required, names))
        tf = true;
        return;
    end
end
tf = false;
end

function ensure_existing_run_file_fields(runFile, A_layers, layer_info, user_ids, buildtag, rngSeed, Theta_full, M)
R = load(runFile);
changed = false;

if ~isfield(R, 'buildtag') || isempty(R.buildtag)
    R.buildtag = buildtag;
    changed = true;
end
if ~isfield(R, 'rngSeed') || isempty(R.rngSeed)
    R.rngSeed = rngSeed;
    changed = true;
end
if ~isfield(R, 'M') || isempty(R.M)
    R.M = M;
    changed = true;
end
if ~isfield(R, 'A_layers') || isempty(R.A_layers)
    R.A_layers = A_layers;
    changed = true;
end
if ~isfield(R, 'layer_info') || isempty(R.layer_info)
    R.layer_info = layer_info;
    changed = true;
end
if ~isfield(R, 'user_ids') || isempty(R.user_ids)
    R.user_ids = user_ids;
    changed = true;
end
if (~isfield(R, 'final_labels') || isempty(R.final_labels)) && ...
        isfield(R, 'active_idx') && isfield(R, 'cluster_labels') && ...
        isfield(R, 'inactive_idx') && isfield(R, 'inactive_results')
    R.final_labels = get_movielens_final_labels(numel(R.A_layers), ...
        R.active_idx, R.cluster_labels, R.inactive_idx, R.inactive_results);
    changed = true;
end
if ~isfield(R, 'Theta_full') || isempty(R.Theta_full)
    R.Theta_full = Theta_full;
    changed = true;
end
if ~isfield(R, 'V_all_embed') || isempty(R.V_all_embed)
    R.V_all_embed = spectral_embed_from_similarity(R.Theta_full, M);
    changed = true;
end

if changed
    if isfield(R, 'beta_star')
        beta_star = R.beta_star; %#ok<NASGU>
    else
        beta_star = NaN; %#ok<NASGU>
    end
    M = R.M; %#ok<NASGU>
    buildtag = R.buildtag; %#ok<NASGU>
    rngSeed = R.rngSeed; %#ok<NASGU>
    A_layers = R.A_layers; %#ok<NASGU>
    layer_info = R.layer_info; %#ok<NASGU>
    user_ids = R.user_ids; %#ok<NASGU>
    if isfield(R, 'final_labels')
        final_labels = R.final_labels; %#ok<NASGU>
    else
        final_labels = []; %#ok<NASGU>
    end
    Theta_full = R.Theta_full; %#ok<NASGU>
    V_all_embed = R.V_all_embed; %#ok<NASGU>
    save(runFile, 'beta_star', 'M', 'buildtag', 'rngSeed', 'A_layers', ...
        'layer_info', 'user_ids', 'final_labels', 'Theta_full', 'V_all_embed', '-append');
end
end
