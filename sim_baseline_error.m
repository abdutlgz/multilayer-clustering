function err = sim_baseline_error(A, label, M, K, L)
% SIM_BASELINE_ERROR
% Pure between-layer clustering baseline with no active/non-active split.

[idcs_all, ~, ~] = BetweenLayerGramm_Gram(A, M, K);

if isempty(idcs_all) || any(isnan(idcs_all))
    err = 1;
    return;
end

trueLab = label(:)';
miss_all = missclassGroups_more(idcs_all(:)', trueLab, M);
err = miss_all / L;
end
