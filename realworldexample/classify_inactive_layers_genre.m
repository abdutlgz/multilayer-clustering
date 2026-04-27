function inactive_results = classify_inactive_layers_genre(A_layers, layer_info, inactive_idx, U_group)
% Classify inactive layers using fitted active-group subspaces.
%
% Input:
%   A_layers    : cell array of rectangular adjacency matrices
%   layer_info  : metadata table
%   inactive_idx: indices of inactive layers
%   U_group     : cell array of fitted left subspaces from active clusters
%
% Output:
%   inactive_results : table with predicted cluster and projection scores

M = numel(U_group);
numInactive = numel(inactive_idx);

pred_cluster = zeros(numInactive,1);
best_score = zeros(numInactive,1);
all_scores = zeros(numInactive, M);

for t = 1:numInactive
    ell = inactive_idx(t);
    A = A_layers{ell};

    scores = -inf(1,M);

    for m = 1:M
        U = U_group{m};

        if isempty(U)
            scores(m) = -inf;
        else
            scores(m) = norm(U' * A, 'fro');
        end
    end

    all_scores(t,:) = scores;
    [best_score(t), pred_cluster(t)] = max(scores);
end

inactive_results = layer_info(inactive_idx, :);
inactive_results.original_layer_id = inactive_idx;
inactive_results.pred_cluster = pred_cluster;
inactive_results.best_score = best_score;

for m = 1:M
    inactive_results.(sprintf('score_group_%d', m)) = all_scores(:,m);
end
end