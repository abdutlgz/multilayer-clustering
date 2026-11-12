function [pred,active]=fit_method(name,A,widths,c)
% One full fit of a method on the network A (cell array of sparse n-by-n_l layers).
% This is the code that is timed in Section 4.3 (bench_one.m) and the UASE-based
% method of Section 4.2.
%   Algorithm1 : active layers n_l >= log(n)/rho_hat (select_active_layers.m),
%                Stage 1 clusters the active layers, Stage 2 assigns each
%                non-active layer to the group m that maximizes
%                ||(Pi_perp U^(m))' A_l||_F^2 (column-centered group subspace).
%   TopXX      : Algorithm 1 with the XX% widest layers active instead of the
%                threshold (XX = 10, 20, ..., 100). Top100 is all-layer clustering.
%   All_layer  : Stage 1 of Algorithm 1 on all layers.
%   UASE       : UASE-based clustering (see uase below).
%   Input_only : does nothing; measures the memory of loading the network.
% Returns the predicted group of every layer and the indices of the active layers.
% c has fields n, K, M, seed. Every random step is seeded from c.seed.
L=numel(A); pred=[]; active=[];
if strncmp(name,'Top',3)
    k=round(str2double(name(4:end))/100*L); assert(k>=1 && k<=L,'bad method %s',name);
    [~,ord]=sort(widths(:),'descend'); active=sort(ord(1:k))';
    pred=two_stage(A,active,c); return
end
switch name
    case 'Input_only'
        return
    case 'UASE'
        pred=uase(A,c); active=1:L;
    case 'All_layer'
        active=1:L; pred=stage1(A,active,c);
    case 'Algorithm1'
        sel=select_active_layers(A,widths,c.n); active=sel.active_ids(:)';
        pred=two_stage(A,active,c);
    otherwise
        error('Unknown method %s',name);
end
end

function pred=two_stage(A,active,c)
% Stage 1 on the active layers, then Stage 2 for the other layers.
L=numel(A); lab=stage1(A,active,c); pred=zeros(L,1); pred(active)=lab;
rest=setdiff(1:L,active); if isempty(rest), return; end
G=cell(1,c.M);
for m=1:c.M
    G{m}=hollow_space(A(active(lab==m)),c.K,c.seed+300000+m);
    G{m}=G{m}-mean(G{m},1);
end
for l=rest
    sc=zeros(1,c.M); for m=1:c.M, sc(m)=norm(G{m}'*A{l},'fro')^2; end
    [~,pred(l)]=max(sc);
end
end

function lab=stage1(A,act,c)
% Stage 1: U_i = leading K eigenvectors of the hollowed Gram matrix of layer i,
% T(i,j) = ||U_i' U_j||_F^2, then k-means on the M leading eigenvectors of T.
N=numel(act); U=cell(1,N);
for i=1:N
    U{i}=hollow_space(A{act(i)},c.K,c.seed+10000+act(i));
end
T=zeros(N);
for i=1:N, for j=1:i, T(i,j)=norm(U{i}'*U{j},'fro')^2; T(j,i)=T(i,j); end, end
[V,D]=eig((T+T')/2,'vector'); [~,ix]=sort(D,'descend');
lab=cluster(V(:,ix(1:c.M)),c);
if numel(act)==numel(A), lab=lab(:); end
end

function pred=uase(A,c)
% UASE-based clustering: V = KM leading left singular vectors of [A_1,...,A_L];
% layer feature = vec(V'(A_l A_l' - diag(A_l A_l'))V), scaled to unit norm;
% spectral clustering of the L features (M leading left singular vectors,
% rows normalized, then k-means).
L=numel(A); unfolded=cat(2,A{:});
[V,~]=leading_space(unfolded,min(c.n,c.K*c.M),'irlba',c.seed+700000); clear unfolded
q=size(V,2); X=zeros(L,q*q);
for l=1:L
    F=V'*A{l}; B=F*F'-V'*(sum(A{l}.^2,2).*V); B=(B+B')/2; x=B(:)';
    if norm(x)>0, x=x/norm(x); end, X(l,:)=x;
end
[W,~]=leading_space(X,c.M,'irlba',c.seed+800000);
pred=cluster(W./max(vecnorm(W,2,2),eps),c);
end

function U=hollow_space(X,k,seed)
% Leading k eigenvectors (largest |eigenvalue|) of the hollowed Gram matrix
% sum_l (X_l*X_l' - diag(X_l*X_l')), computed with matrix-vector products only,
% so the dense n-by-n matrix is never formed. X is one layer or a cell array of
% layers (a group in Stage 2); the layers are not copied into one matrix.
if ~iscell(X), X={X}; end
n=size(X{1},1); d=zeros(n,1);
for i=1:numel(X), d=d+full(sum(X{i}.^2,2)); end
rng(seed,'twister');
if n<=k+5
    H=zeros(n); for i=1:numel(X), H=H+full(X{i}*X{i}'); end
    H(1:n+1:end)=0; [U,~,~]=svd(H); U=U(:,1:k); return
end
[U,~,flag]=eigs(@(x) gram_times(X,x)-d.*x,n,k,'largestabs','Tolerance',1e-7, ...
    'MaxIterations',5000,'StartVector',randn(n,1),'IsFunctionSymmetric',true);
assert(flag==0,'eigs did not converge'); assert(norm(U'*U-eye(k),'fro')<1e-5);
end

function y=gram_times(X,x)
y=X{1}*(X{1}'*x);
for i=2:numel(X), y=y+X{i}*(X{i}'*x); end
end

function lab=cluster(W,c)
rng(c.seed+400000,'twister');
lab=kmeans(W,c.M,'Replicates',10,'MaxIter',1000,'EmptyAction','singleton');
end
