function err = sim_split_pipeline_error(A, label, idx_active, idx_nonactive, M, K, n, L)
% SIM_SPLIT_PIPELINE_ERROR
% Runs the split-and-classify procedure and returns total misclassification
% error. Failure cases are assigned error 1.

if isempty(idx_active) || numel(idx_active) < M
    err = 1;
    return;
end

A_active = A(idx_active);
[idcs_active, ~, ~] = BetweenLayerGramm_Gram(A_active, M, K);

if isempty(idcs_active) || any(isnan(idcs_active))
    err = 1;
    return;
end

s_hat = zeros(1, L);
s_hat(idx_active) = idcs_active(:)';

Uhat_group = cell(1, M);
for m = 1:M
    layers_m = idx_active(idcs_active == m);

    if isempty(layers_m)
        err = 1;
        return;
    end

    Hm = zeros(n, n);
    for ll = layers_m(:)'
        G = A{ll} * A{ll}';
        G(1:n+1:end) = 0;
        Hm = Hm + G;
    end

    try
        [Uhat, ~] = eigs(Hm, K, 'largestreal');
    catch
        err = 1;
        return;
    end
    Uhat_group{m} = Uhat;
end

for ll = idx_nonactive
    scores = zeros(1, M);
    for m = 1:M
        scores(m) = norm(Uhat_group{m}' * A{ll}, 'fro')^2;
    end
    [~, s_hat(ll)] = max(scores);
end

trueLab = label(:)';
miss_all = missclassGroups_more(s_hat, trueLab, M);
err = miss_all / L;
end
