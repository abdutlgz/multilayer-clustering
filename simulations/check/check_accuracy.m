function ok=check_accuracy(id, reps)
% Reruns the first `reps` networks of accuracy setting `id` with the code in this
% repository and compares them with the stored results in results/accuracy.
%   check_accuracy            % setting 72 (n=250, L=150, rho=0.10), 10 networks, ~10 s
%   check_accuracy(72, 100)   % all 100 networks, ~1.5 min; also compares the
%                             % fixed-threshold curve with curves.csv
% Prints the largest difference and returns true if everything matches (up to the
% 6 decimals stored in the CSV files).
% The stored results were computed with MATLAB R2023a. In settings where the
% signal is too weak for any method (errors near 60%, e.g. setting 1: n=100,
% rho=0.05), k-means can end in a different partition on another MATLAB version,
% so single networks may differ there; the code itself gives identical results.
if nargin<1, id=72; end
if nargin<2, reps=10; end
here=fileparts(mfilename('fullpath')); root=fullfile(here,'..');
addpath(fullfile(root,'accuracy'),fullfile(root,'lib'));
tmp=fullfile(tempdir,sprintf('check_accuracy_%d_%d',id,feature('getpid'))); mkdir(tmp);
cleanup=onCleanup(@() rmdir(tmp,'s'));
run_setting(id,tmp,reps);
q=load(fullfile(tmp,sprintf('setting_%03d.mat',id))); R=q.R;
N=readtable(fullfile(root,'results','accuracy','networks.csv'));
N=N(N.id==id & N.rep<=reps,:); N=sortrows(N,'rep');
new=[[R.rule_count]' cat(2,R.rule)' [R.all_layer]' [R.uase]'];
old=[N.rule_count N.rule_err N.rule_err_active N.rule_err_nonactive N.all_layer_err N.uase_err];
d=abs(new-old); d(isnan(new)&isnan(old))=0;
names={'active layers','Algorithm 1 total','clustering','classification','all-layer','UASE'};
fprintf('\nSetting %d, networks 1-%d: largest difference from results/accuracy/networks.csv\n',id,reps);
for k=1:numel(names), fprintf('  %-18s %.2e\n',names{k},max(d(:,k))); end
ok=max(d(:))<1e-5;
if reps==100
    E=cat(3,R.beta_err); E=reshape(E(1,:,:),numel(q.meta.betas),reps); mu=mean(E,2,'omitnan');
    C=readtable(fullfile(root,'results','accuracy','curves.csv')); C=sortrows(C(C.id==id,:),'beta');
    dc=abs(mu-C.mean_err); dc(isnan(mu)&isnan(C.mean_err))=0;
    fprintf('  %-18s %.2e\n','Figure 1 curve',max(dc)); ok=ok && max(dc)<1e-5;
end
if ok, fprintf('MATCH\n'); else, fprintf('DIFFERENT\n'); end
end
