function uaseRes = cluster_uase_layers(A_layers, trueLabels, M, K)
%CLUSTER_UASE_LAYERS Unfolded adjacency spectral embedding baseline.
%
% This implements the UASE idea directly: concatenate rectangular layers
% horizontally, estimate a common left embedding, and cluster layer features
% obtained by projecting each hollowed Gram matrix into that embedding.

try
    L = numel(A_layers);
    n = size(A_layers{1}, 1);
    d = min(n, max(K * M, K));
    totalCols = sum(cellfun(@(A) size(A, 2), A_layers));
    A_unfold = sparse(n, totalCols);
    colStart = 1;
    for ell = 1:L
        nl = size(A_layers{ell}, 2);
        cols = colStart:(colStart + nl - 1);
        A_unfold(:, cols) = A_layers{ell};
        colStart = colStart + nl;
    end

    U = top_left_singular_vectors(A_unfold, d);
    layerFeatures = zeros(L, d * d);
    for ell = 1:L
        G = full(A_layers{ell} * A_layers{ell}');
        G(1:n+1:end) = 0;
        G = (G + G') / 2;
        B = U' * G * U;
        feat = reshape(B, 1, []);
        nf = norm(feat);
        if nf > 0
            feat = feat / nf;
        end
        layerFeatures(ell,:) = feat;
    end

    embed = top_left_singular_vectors(layerFeatures, M);
    embed = normalize_rows(embed);
    pred = kmeans(embed, M, 'Replicates', 20, 'MaxIter', 1000);
    if nargin >= 2 && ~isempty(trueLabels)
        err = layer_misclassification_local(pred, trueLabels, M) / numel(trueLabels);
    else
        err = NaN;
    end
    status = 'ok';
catch ME
    pred = zeros(numel(A_layers),1);
    err = 1;
    status = ME.identifier;
end

uaseRes = struct('pred', pred, 'error', err, 'status', status);
end

function miss = layer_misclassification_local(predLabels, trueLabels, M)
predLabels = predLabels(:)';
trueLabels = trueLabels(:)';
permsM = perms(1:M);
missVals = zeros(size(permsM,1),1);
for p = 1:size(permsM,1)
    relabeled = permsM(p, trueLabels);
    missVals(p) = sum(predLabels ~= relabeled);
end
miss = min(missVals);
end
