function run_setting(id, outdir, reps)
% Section 4.2 (accuracy), one setting: row `id` of settings.csv, `reps` networks
% (default 100). For each network it records
%   - Algorithm 1 with the threshold n_l >= log(n)/rho_hat (eq. sim:threshold),
%   - Algorithm 1 with every fixed threshold beta = 0.30, 0.35, ..., 1.10
%     (active layers: log(n_l)/log(n) >= beta), for Figure 1,
%   - all-layer clustering and the UASE-based method, for Figure 2.
% Errors of Algorithm 1 are stored as [total; active; non-active], i.e. the total,
% clustering and classification errors of Table 1.
% Network r of setting id uses seed 6000000 + 1000*id + r.
% Results go to outdir/setting_###.mat after every network, so a rerun continues
% where it stopped.
if nargin<3, reps=100; end
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','lib'));
maxNumCompThreads(1);
S=readmatrix(fullfile(here,'settings.csv')); row=S(S(:,1)==id,:); assert(size(row,1)==1);
if ~isfolder(outdir), mkdir(outdir); end
betas=0.30:0.05:1.10; out=fullfile(outdir,sprintf('setting_%03d.mat',id));
R=struct('rep',{},'seed',{},'rho_hat',{},'rule_beta',{},'rule_beta_eff',{},'rule_count',{}, ...
         'groups_active',{},'rule',{},'beta_err',{},'beta_count',{},'all_layer',{},'uase',{},'seconds',{});
if isfile(out), q=load(out,'R'); R=q.R; end
for r=numel(R)+1:reps
    t0=tic;
    c=struct('n',row(2),'L',row(3),'rho',row(4),'K',3,'M',3,'kappa',10,'seed',6000000+1000*id+r);
    [A,truth,widths]=generate_network(c); A=cellfun(@double,A,'UniformOutput',false);
    % Stage 1 similarity matrix T over all layers; every active set uses a sub-block.
    U=cell(1,c.L);
    for l=1:c.L, [U{l},~]=leading_space(hollow(A{l}),c.K,'eigs',c.seed+10000+l); end
    T=zeros(c.L);
    for i=1:c.L, for j=1:i, T(i,j)=norm(U{i}'*U{j},'fro')^2; T(j,i)=T(i,j); end, end
    clear U
    sel=select_active_layers(A,widths,c.n);
    rule=algorithm1(A,T,sel.active_ids,c,truth);
    beta_err=nan(3,numel(betas)); beta_count=zeros(1,numel(betas));
    for b=1:numel(betas)
        act=find(log(widths)/log(c.n)>=betas(b)-1e-12); beta_count(b)=numel(act);
        beta_err(:,b)=algorithm1(A,T,act,c,truth);
    end
    [V,D]=eig((T+T')/2,'vector'); [~,ix]=sort(D,'descend');
    all_layer=layer_errors(cluster(V(:,ix(1:c.M)),c),truth,c.M,1:c.L);
    uase=layer_errors(fit_method('UASE',A,widths,c),truth,c.M,1:c.L);
    R(r)=struct('rep',r,'seed',c.seed,'rho_hat',sel.rho_hat,'rule_beta',sel.beta_formula, ...
        'rule_beta_eff',sel.beta_effective,'rule_count',sel.active_count, ...
        'groups_active',numel(unique(truth(sel.active_ids))),'rule',rule, ...
        'beta_err',beta_err,'beta_count',beta_count,'all_layer',all_layer,'uase',uase,'seconds',toc(t0));
    meta=struct('id',id,'n',c.n,'L',c.L,'rho',c.rho,'K',c.K,'M',c.M,'kappa',c.kappa,'betas',betas, ...
        'note','rule and beta_err rows: [total; active; nonactive] error of Algorithm 1');
    tmp=[out '.tmp.mat']; save(tmp,'R','meta'); movefile(tmp,out);
    fprintf('setting %d n=%d L=%d rho=%.2f rep %d: active %d, Algorithm 1 %.3f, all-layer %.3f, UASE %.3f (%.1fs)\n', ...
        id,c.n,c.L,c.rho,r,sel.active_count,rule(1),all_layer,uase,toc(t0));
end
end

function e=algorithm1(A,T,act,c,truth)
% Algorithm 1 for a given active set: Stage 1 clusters the active layers using
% T(act,act); Stage 2 assigns each other layer to the group m maximizing
% ||(Pi_perp U^(m))' A_l||_F^2. Returns [total; active; nonactive] error,
% or NaN if fewer than M layers are active.
act=act(:)'; L=numel(A); e=nan(3,1);
if numel(act)<c.M, return; end
[V,D]=eig((T(act,act)+T(act,act)')/2,'vector'); [~,ix]=sort(D,'descend');
lab=cluster(V(:,ix(1:c.M)),c); pred=zeros(L,1); pred(act)=lab; G=cell(1,c.M);
for m=1:c.M
    X=[A{act(lab==m)}]; H=full(X*X'); H(1:c.n+1:end)=0;
    [G{m},~]=leading_space(H,c.K,'eigs',c.seed+300000+m);
    G{m}=G{m}-mean(G{m},1);
end
for l=setdiff(1:L,act)
    sc=zeros(1,c.M); for m=1:c.M, sc(m)=norm(G{m}'*A{l},'fro')^2; end
    [~,pred(l)]=max(sc);
end
[e(1),e(2),e(3)]=layer_errors(pred,truth,c.M,act);
end

function lab=cluster(W,c)
rng(c.seed+400000,'twister');
lab=kmeans(W,c.M,'Replicates',10,'MaxIter',1000,'EmptyAction','singleton');
end

function H=hollow(A)
H=full(A*A'); H(1:size(H,1)+1:end)=0;
end
