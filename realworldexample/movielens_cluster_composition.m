function compTbl = movielens_cluster_composition(layer_info, final_labels, M, selectedGenres)
%MOVIELENS_CLUSTER_COMPOSITION Count final layer assignments by genre.

if nargin < 4 || isempty(selectedGenres)
    selectedGenres = unique(cellstr(string(layer_info.genre)), 'stable');
end
if nargin < 3 || isempty(M)
    M = max(final_labels);
end

genres = cellstr(selectedGenres(:));
counts = zeros(numel(genres), M);
layerGenres = string(layer_info.genre);

for g = 1:numel(genres)
    idx_g = layerGenres == string(genres{g});
    for m = 1:M
        counts(g,m) = sum(idx_g & final_labels(:) == m);
    end
end

compTbl = array2table(counts);
for m = 1:M
    compTbl.Properties.VariableNames{m} = sprintf('Cluster_%d', m);
end
compTbl = addvars(compTbl, genres, 'Before', 1, 'NewVariableNames', 'Genre');
end
