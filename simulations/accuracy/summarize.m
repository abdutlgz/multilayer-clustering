function summarize(outdir)
% Collect outdir/setting_###.mat into three CSV files in outdir/summary:
%   curves.csv           mean error of Algorithm 1 at each fixed threshold beta (Figure 1)
%   settings_summary.csv one row per setting: Algorithm 1 with the data-driven
%                        threshold, best fixed threshold, all-layer and UASE (Table 1, Figure 2)
%   networks.csv         one row per network
% Errors are fractions of layers (multiply by 100 for %).
here=fileparts(mfilename('fullpath')); S=readmatrix(fullfile(here,'settings.csv'));
sd=fullfile(outdir,'summary'); if ~isfolder(sd), mkdir(sd); end
fc=fopen(fullfile(sd,'curves.csv'),'w'); fprintf(fc,'id,n,L,rho,beta,mean_err,se_err,n_valid,mean_active\n');
fs=fopen(fullfile(sd,'settings_summary.csv'),'w');
fprintf(fs,['id,n,L,rho,reps,rule_err,rule_se,rule_err_active,rule_err_nonactive,rule_beta,rule_count,' ...
    'n_invalid,frac_all_groups,best_beta,best_err,all_layer_err,all_layer_se,uase_err,uase_se,mean_seconds\n']);
fn=fopen(fullfile(sd,'networks.csv'),'w');
fprintf(fn,['id,n,L,rho,rep,seed,rho_hat,rule_beta,rule_beta_eff,rule_count,groups_active,' ...
    'rule_err,rule_err_active,rule_err_nonactive,all_layer_err,uase_err,seconds\n']);
missing=[];
for i=1:size(S,1)
    id=S(i,1); f=fullfile(outdir,sprintf('setting_%03d.mat',id));
    if ~isfile(f), missing(end+1)=id; continue; end %#ok<AGROW>
    q=load(f); R=q.R; m=q.meta; k=numel(R);
    E=cat(3,R.beta_err); E=reshape(E(1,:,:),numel(m.betas),k);   % betas x reps, total error
    C=cat(1,R.beta_count);
    mu=mean(E,2,'omitnan'); nv=sum(~isnan(E),2); se=std(E,0,2,'omitnan')./sqrt(nv);
    for b=1:numel(m.betas)
        fprintf(fc,'%d,%d,%d,%.2f,%.2f,%.6f,%.6f,%d,%.2f\n',id,m.n,m.L,m.rho,m.betas(b),mu(b),se(b),nv(b),mean(C(:,b)));
    end
    rule=cat(2,R.rule); ok=~isnan(rule(1,:)); [be,bi]=min(mu); al=[R.all_layer]; ua=[R.uase];
    fprintf(fs,'%d,%d,%d,%.2f,%d,%.6f,%.6f,%.6f,%.6f,%.4f,%.2f,%d,%.3f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f\n', ...
        id,m.n,m.L,m.rho,k,mean(rule(1,ok)),std(rule(1,ok))/sqrt(sum(ok)),mean(rule(2,ok)), ...
        mean(rule(3,ok),'omitnan'),mean([R.rule_beta]),mean([R.rule_count]),sum(~ok), ...
        mean([R.groups_active]==m.M),m.betas(bi),be,mean(al),std(al)/sqrt(k),mean(ua),std(ua)/sqrt(k),mean([R.seconds]));
    for r=1:k
        x=R(r);
        fprintf(fn,'%d,%d,%d,%.2f,%d,%d,%.6f,%.4f,%.4f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f\n',id,m.n,m.L,m.rho, ...
            x.rep,x.seed,x.rho_hat,x.rule_beta,x.rule_beta_eff,x.rule_count,x.groups_active, ...
            x.rule(1),x.rule(2),x.rule(3),x.all_layer,x.uase,x.seconds);
    end
end
fclose(fc); fclose(fs); fclose(fn);
fprintf('Summarized %d of %d settings into %s. Missing: %s\n',size(S,1)-numel(missing),size(S,1),sd,mat2str(missing));
end
