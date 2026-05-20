function X = normalize_rows(X)
%NORMALIZE_ROWS Scale rows to unit Euclidean norm, leaving zero rows fixed.

rowNorms = sqrt(sum(X.^2, 2));
rowNorms(rowNorms == 0) = 1;
X = X ./ rowNorms;
end
