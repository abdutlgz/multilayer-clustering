function U = top_left_singular_vectors(X, r)
%TOP_LEFT_SINGULAR_VECTORS Robust leading-left-singular-vector helper.

r = min([r, size(X,1), size(X,2)]);
if r <= 0
    U = zeros(size(X,1), 0);
    return;
end

try
    if (issparse(X) || max(size(X)) > 500) && r < min(size(X))
        opts = struct();
        opts.maxit = 1000;
        opts.tol = 1e-8;
        opts.disp = 0;
        [U,~,~] = svds(X, r, 'largest', opts);
    else
        [Ufull,Sfull,~] = svd(full(X), 'econ');
        [~,ord] = sort(diag(Sfull), 'descend');
        U = Ufull(:, ord(1:r));
    end
catch
    [Ufull,Sfull,~] = svd(full(X), 'econ');
    [~,ord] = sort(diag(Sfull), 'descend');
    U = Ufull(:, ord(1:r));
end

U = real(U);
end
