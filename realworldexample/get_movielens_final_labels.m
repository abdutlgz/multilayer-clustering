function final_labels = get_movielens_final_labels(numLayers, active_idx, cluster_labels, inactive_idx, inactive_results)
%GET_MOVIELENS_FINAL_LABELS Combine active clustering and inactive predictions.

final_labels = zeros(numLayers, 1);
final_labels(active_idx) = cluster_labels(:);

if ~isempty(inactive_idx)
    if isempty(inactive_results) || ~ismember('pred_cluster', inactive_results.Properties.VariableNames)
        error('get_movielens_final_labels:MissingPredictions', ...
            'inactive_results.pred_cluster is required when inactive_idx is nonempty.');
    end
    final_labels(inactive_idx) = inactive_results.pred_cluster(:);
end
end
