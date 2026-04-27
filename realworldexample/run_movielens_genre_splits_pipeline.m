%% run_movielens_genre_splits_pipeline.m
clear; clc;

load('movielens_genre_split_layers.mat', 'A_layers', 'layer_info', 'user_ids');

resultsDir = 'results_movielens';
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

rngSeed = 7;                % reproducibility

%% settings
%beta_star = 0.7;          % adjust after seeing counts
%beta_star = 0.75;  
beta_star = 0.8;  
M = 3;                     % try 3 first
rank_method = 'energy';
energy_thresh = 0.90;
fixed_rank = 5;
maxRankTry = 10;
use_normalized_similarity = true;

fprintf('Selecting active layers...\n');
active_idx = find(layer_info.beta_l >= beta_star);
inactive_idx = find(layer_info.beta_l < beta_star);

fprintf('Total layers   : %d\n', numel(A_layers));
fprintf('Active layers  : %d\n', numel(active_idx));
fprintf('Inactive layers: %d\n', numel(inactive_idx));

if isempty(active_idx)
    error('No active layers for this beta_star.');
end

A_active = A_layers(active_idx);
info_active = layer_info(active_idx,:);

%% layerwise SVD
numActive = numel(A_active);
U_active = cell(numActive,1);
Khat_active = zeros(numActive,1);

for l = 1:numActive
    A = A_active{l};

    r_try = min([maxRankTry, size(A,1)-1, size(A,2)-1]);
    r_try = max(r_try, 1);

    [U0,S0,~] = svds(A, r_try);
    singVals = diag(S0).^2;

    switch lower(rank_method)
        case 'fixed'
            Khat = min(fixed_rank, r_try);
        case 'energy'
            singVals(singVals < 0) = 0;
            if sum(singVals) <= 0
                Khat = 1;
            else
                cumEnergy = cumsum(singVals) / sum(singVals);
                Khat = find(cumEnergy >= energy_thresh, 1, 'first');
                if isempty(Khat)
                    Khat = min(5, r_try);
                end
            end
        otherwise
            error('Unknown rank_method.');
    end

    Khat = max(1, Khat);

    if Khat < r_try
        [U,~,~] = svds(A, Khat);
    else
        U = U0(:,1:Khat);
    end

    U_active{l} = U;
    Khat_active(l) = Khat;
end

%% similarity matrix
Theta_hat = zeros(numActive, numActive);

for i = 1:numActive
    Ui = U_active{i};
    ki = size(Ui,2);

    for j = i:numActive
        Uj = U_active{j};
        kj = size(Uj,2);

        val = norm(Ui' * Uj, 'fro')^2;
        if use_normalized_similarity
            val = val / min(ki, kj);
        end

        Theta_hat(i,j) = val;
        Theta_hat(j,i) = val;
    end
end

%% spectral clustering
Theta_sym = (Theta_hat + Theta_hat') / 2;
[V_M,~] = eigs(Theta_sym, M, 'largestreal');

rowNorms = sqrt(sum(V_M.^2,2));
rowNorms(rowNorms == 0) = 1;
V_embed = V_M ./ rowNorms;

cluster_labels = kmeans(V_embed, M, 'Replicates', 30, 'MaxIter', 1000);

%% build group subspaces from active layers
fprintf('Estimating group-level subspaces from active layers...\n');

U_group = cell(M,1);
Khat_group = zeros(M,1);

for m = 1:M
    idx_m = find(cluster_labels == m);

    if isempty(idx_m)
        warning('Cluster %d is empty.', m);
        U_group{m} = [];
        Khat_group(m) = 0;
        continue;
    end

    % choose group rank from median active rank
    K_group = max(1, round(median(Khat_active(idx_m))));
    Khat_group(m) = K_group;

    % aggregate active layers in this cluster by horizontal concatenation
    A_concat = [];
    for t = 1:numel(idx_m)
        A_concat = [A_concat, A_active{idx_m(t)}]; %#ok<AGROW>
    end

    [U_m,~,~] = svds(A_concat, K_group);
    U_group{m} = U_m;

    fprintf('Group %d | active layers = %d | K_group = %d\n', ...
        m, numel(idx_m), K_group);
end
%% classify inactive layers
if ~isempty(inactive_idx)
    fprintf('Classifying inactive layers...\n');
    inactive_results = classify_inactive_layers_genre(A_layers, layer_info, inactive_idx, U_group);

    disp(sortrows(inactive_results(:, ...
        {'original_layer_id','genre','split_id','n_movies','n_edges','beta_l','pred_cluster','best_score'}), ...
        {'pred_cluster','genre','split_id'}));

    summary_inactive = groupsummary(inactive_results, {'pred_cluster','genre'});
    disp(summary_inactive(:, {'pred_cluster','genre','GroupCount'}));
else
    inactive_results = table();
    fprintf('No inactive layers to classify.\n');
end
%% final labels for all layers
numLayers = numel(A_layers);
final_labels = zeros(numLayers,1);

% active layers keep their clustering labels
final_labels(active_idx) = cluster_labels;

% inactive layers get classified labels
if ~isempty(inactive_idx)
    final_labels(inactive_idx) = inactive_results.pred_cluster;
end
%% results
active_results = info_active;
active_results.original_layer_id = active_idx;
active_results.Khat = Khat_active;
active_results.cluster = cluster_labels;

disp(sortrows(active_results(:, {'original_layer_id','genre','split_id','n_movies','n_edges','beta_l','cluster'}), ...
    {'cluster','genre','split_id'}));

summary_genre = groupsummary(active_results, {'cluster','genre'});
disp(summary_genre(:, {'cluster','genre','GroupCount'}));

%% compute layerwise subspaces for ALL layers
numLayers = numel(A_layers);
U_all = cell(numLayers,1);
Khat_all = zeros(numLayers,1);

for l = 1:numLayers
    A = A_layers{l};

    r_try = min([maxRankTry, size(A,1)-1, size(A,2)-1]);
    r_try = max(r_try, 1);

    [U0,S0,~] = svds(A, r_try);
    singVals = diag(S0).^2;

    switch lower(rank_method)
        case 'fixed'
            Khat = min(fixed_rank, r_try);

        case 'energy'
            singVals(singVals < 0) = 0;
            if sum(singVals) <= 0
                Khat = 1;
            else
                cumEnergy = cumsum(singVals) / sum(singVals);
                Khat = find(cumEnergy >= energy_thresh, 1, 'first');
                if isempty(Khat)
                    Khat = min(5, r_try);
                end
            end

        otherwise
            error('Unknown rank_method.');
    end

    Khat = max(1, Khat);

    if Khat < r_try
        [U,~,~] = svds(A, Khat);
    else
        U = U0(:,1:Khat);
    end

    U_all{l} = U;
    Khat_all(l) = Khat;
end
%% full similarity matrix for all layers
Theta_full = zeros(numLayers, numLayers);

for i = 1:numLayers
    Ui = U_all{i};
    ki = size(Ui,2);

    for j = i:numLayers
        Uj = U_all{j};
        kj = size(Uj,2);

        val = norm(Ui' * Uj, 'fro')^2;

        if use_normalized_similarity
            val = val / min(ki, kj);
        end

        Theta_full(i,j) = val;
        Theta_full(j,i) = val;
    end
end
%% spectral embedding for all layers
Theta_full_sym = (Theta_full + Theta_full') / 2;
[V_all,~] = eigs(Theta_full_sym, M, 'largestreal');

rowNorms = sqrt(sum(V_all.^2,2));
rowNorms(rowNorms == 0) = 1;
V_all_embed = V_all ./ rowNorms;
%% plot all layers
figure;
if M >= 2
    gscatter(V_all_embed(:,1), V_all_embed(:,2), final_labels, [], 'o', 8);
    xlabel('Coord 1');
    ylabel('Coord 2');
    title('All-layer embedding with final labels', 'Interpreter', 'none');
else
    scatter(1:numLayers, V_all_embed(:,1), 50, final_labels, 'filled');
    xlabel('Layer index');
    ylabel('Coord 1');
    title('All-layer embedding with final labels', 'Interpreter', 'none');
end
%% plot all layers: active vs inactive
figure; hold on;

colors = lines(M);

for m = 1:M
    idx_active_m = intersect(active_idx, find(final_labels == m));
    idx_inactive_m = intersect(inactive_idx, find(final_labels == m));

    scatter(V_all_embed(idx_active_m,1), V_all_embed(idx_active_m,2), ...
        70, colors(m,:), 'filled', 'o');

    scatter(V_all_embed(idx_inactive_m,1), V_all_embed(idx_inactive_m,2), ...
        80, colors(m,:), 'x', 'LineWidth', 1.8);
end

xlabel('Coord 1');
ylabel('Coord 2');
title('All-layer embedding: active (filled) vs inactive (x)', 'Interpreter', 'none');
grid on; box on;
hold off;
%% reordered full heatmap
[~,ord] = sort(final_labels);
Theta_full_ord = Theta_full(ord, ord);

figure;
imagesc(Theta_full_ord);
colorbar;
title('Theta-full reordered by final labels', 'Interpreter', 'none');

saveFileName = fullfile(resultsDir, sprintf( ...
    'movielens_run_users%d_splits%d_beta%.2f_M%d_seed%d.mat', ...
    numel(user_ids), max(layer_info.split_id), beta_star, M, rngSeed));

save(saveFileName, ...
    'beta_star', 'M', 'rngSeed', ...
    'A_layers', ...
    'active_idx', 'inactive_idx', ...
    'active_results', 'inactive_results', ...
    'Theta_hat', 'cluster_labels', ...
    'U_group', 'Khat_group', ...
    'layer_info', 'user_ids', ...
    '-v7.3');