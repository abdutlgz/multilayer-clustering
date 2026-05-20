function hooiRes = cluster_hooi_layers(A_layers, trueLabels, M, K)
%CLUSTER_HOOI_LAYERS Tensor baseline on hollowed left Gram layers.
%
% Rectangular layers are converted to common n-by-n hollowed Gram matrices
% and stacked as a third-order tensor. A compact HOOI iteration estimates a
% shared node factor and a layer factor; rows of the layer factor are then
% clustered into M groups.

try
    L = numel(A_layers);
    n = size(A_layers{1}, 1);
    rNode = min(n, max(K * M, K));
    X = zeros(n, n, L);
    for ell = 1:L
        G = full(A_layers{ell} * A_layers{ell}');
        G(1:n+1:end) = 0;
        G = (G + G') / 2;
        nf = norm(G, 'fro');
        if nf > 0
            G = G / nf;
        end
        X(:,:,ell) = G;
    end

    Y3 = reshape(permute(X, [3 1 2]), L, n * n);
    V = top_left_singular_vectors(Y3, M);

    S0 = zeros(n, n);
    for ell = 1:L
        S0 = S0 + X(:,:,ell) * X(:,:,ell)';
    end
    U = top_left_singular_vectors(S0, rNode);

    maxIter = 12;
    for iter = 1:maxIter %#ok<NASGU>
        Z = zeros(n, rNode * M);
        for j = 1:M
            Sj = zeros(n, n);
            for ell = 1:L
                Sj = Sj + V(ell,j) * X(:,:,ell);
            end
            cols = (j-1) * rNode + (1:rNode);
            Z(:,cols) = Sj * U;
        end
        U = top_left_singular_vectors(Z, rNode);

        layerFeatures = zeros(L, rNode * rNode);
        for ell = 1:L
            B = U' * X(:,:,ell) * U;
            layerFeatures(ell,:) = reshape(B, 1, []);
        end
        V = top_left_singular_vectors(layerFeatures, M);
    end

    embed = normalize_rows(V);
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

hooiRes = struct('pred', pred, 'error', err, 'status', status);
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
