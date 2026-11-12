function [U,info]=leading_space(X,k,solver,seed)
% Leading k left singular vectors of X. For a symmetric X with solver 'eigs'
% these are the eigenvectors of the k largest |eigenvalues|.
% solver: 'irlba' (any X) or 'eigs' (square symmetric X). seed fixes the start vector.
rng(seed,'twister'); info=struct('solver',solver,'flag',0,'matvecs',0,'dense_small',false);
assert(k<=min(size(X)),'leading_space:rank','Rank exceeds matrix dimensions.');
if min(size(X))<=k+5
    [U,~,~]=svd(X,'econ'); U=U(:,1:k); info.dense_small=true; return
end
if strcmp(solver,'irlba')
    opts=struct('K',k,'ADJUST',3,'M',min(min(size(X))-1,max(20,2*k+5)), ...
        'MAXIT',5000,'TOL',1e-7,'SIGMA','LS','REORTH','TWO');
    [U,~,~,flag]=irlba(X,opts); info.flag=flag(1); info.matvecs=flag(2);
elseif strcmp(solver,'eigs')
    assert(isequal(size(X,1),size(X,2)),'eigs requires a square symmetric matrix');
    opts=struct('tol',1e-7,'maxit',5000,'disp',0,'v0',randn(size(X,1),1));
    [U,~,flag]=eigs(X,k,'largestabs',opts); info.flag=flag;
else
    error('leading_space:solver','Unknown solver');
end
if info.flag~=0, error('leading_space:nonconvergence','%s failed with flag %d',solver,info.flag); end
assert(all(isfinite(U(:))) && norm(U'*U-eye(k),'fro')<1e-5,'leading_space:orthogonality','Invalid subspace');
end
