function [A,truth,widths,info]=generate_network(c)
% One simulated multilayer network (Section 4.1 of the paper).
%   widths      : n_l = max(K, floor(n^beta_l)), beta_l ~ Unif[0.30, 1.10]
%   groups      : each layer is assigned to one of M groups at random (truth)
%   B^(l)       : 2*rho*U, U ~ Unif(0,1)^{KxK} (not symmetrized), drawn again
%                 while sigma_1(U)/sigma_K(U) > c.kappa (default 10)
%   communities : rows of each group and columns of each layer are split into K
%                 communities at random, with sizes that differ by at most one
%   A{l}        : sparse n-by-n_l matrix, A{l}(i,j) ~ Bernoulli(P^(l)(i,j))
% Fields of c: n, L, K, M, rho, seed, kappa (optional).
if ~isfield(c,'kappa'), c.kappa=10; end
rng(c.seed,'twister'); n=c.n; L=c.L; K=c.K; M=c.M;
widths=max(K,floor(n.^(0.30+0.80*rand(1,L))));
ZX=zeros(n,K,M);
for m=1:M, ZX(:,:,m)=full(sparse(1:n,balanced(n,K),1,n,K)); end
truth=randi(M,1,L); A=cell(1,L); ratio=zeros(1,L); tries=zeros(1,L);
for l=1:L
    nl=widths(l); ZY=full(sparse(1:nl,balanced(nl,K),1,nl,K));
    while true
        U=rand(K,K); s=svd(U); tries(l)=tries(l)+1;
        if s(1)/s(K)<=c.kappa, break; end
    end
    b=2*c.rho*U; ratio(l)=s(1)/s(K);
    A{l}=sparse(binornd(1,ZX(:,:,truth(l))*b*ZY'));
end
info=struct('ratio',ratio,'tries',tries);
end

function g=balanced(N,K)
% Random community labels 1..K with sizes that differ by at most one.
g=mod(randperm(N),K)'+1;
end
