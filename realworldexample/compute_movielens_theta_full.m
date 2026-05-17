function [Theta_full, V_all_embed, U_all, Khat_all] = compute_movielens_theta_full(A_layers, varargin)
%COMPUTE_MOVIELENS_THETA_FULL Compute all-layer subspace similarities.

p = inputParser;
addParameter(p, 'M', []);
addParameter(p, 'rank_method', 'energy');
addParameter(p, 'energy_thresh', 0.90);
addParameter(p, 'fixed_rank', 5);
addParameter(p, 'maxRankTry', 10);
addParameter(p, 'use_normalized_similarity', true);
parse(p, varargin{:});
opts = p.Results;

if isempty(A_layers)
    error('compute_movielens_theta_full:MissingLayers', 'A_layers is empty.');
end

numLayers = numel(A_layers);
U_all = cell(numLayers,1);
Khat_all = zeros(numLayers,1);

for l = 1:numLayers
    A = A_layers{l};
    if isempty(A)
        error('compute_movielens_theta_full:EmptyLayer', 'A_layers{%d} is empty.', l);
    end

    r_try = min([opts.maxRankTry, size(A,1)-1, size(A,2)-1]);
    r_try = max(r_try, 1);

    [U0,S0,~] = svds(A, r_try);
    singVals = diag(S0).^2;

    switch lower(opts.rank_method)
        case 'fixed'
            Khat = min(opts.fixed_rank, r_try);
        case 'energy'
            singVals(singVals < 0) = 0;
            if sum(singVals) <= 0
                Khat = 1;
            else
                cumEnergy = cumsum(singVals) / sum(singVals);
                Khat = find(cumEnergy >= opts.energy_thresh, 1, 'first');
                if isempty(Khat)
                    Khat = min(5, r_try);
                end
            end
        otherwise
            error('compute_movielens_theta_full:BadRankMethod', ...
                'Unknown rank_method: %s', opts.rank_method);
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

Theta_full = zeros(numLayers, numLayers);
for i = 1:numLayers
    Ui = U_all{i};
    ki = size(Ui,2);
    for j = i:numLayers
        Uj = U_all{j};
        kj = size(Uj,2);
        val = norm(Ui' * Uj, 'fro')^2;
        if opts.use_normalized_similarity
            val = val / min(ki, kj);
        end
        Theta_full(i,j) = val;
        Theta_full(j,i) = val;
    end
end

if isempty(opts.M)
    V_all_embed = [];
else
    V_all_embed = movielens_spectral_embed(Theta_full, opts.M);
end
end

function V_embed = movielens_spectral_embed(Theta, M)
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
