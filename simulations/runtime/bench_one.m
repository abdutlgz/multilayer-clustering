function bench_one(netfile, method, outfile)
% Times one method on one saved network. Meant to run in its own MATLAB process,
% so that the peak memory of the process (measured outside, by /usr/bin/time)
% belongs to this method only. Loads the network, warms up on a small network,
% then runs 3 full fits; the reported time is their mean. For Algorithm 1 the
% time includes computing the threshold.
% Writes one CSV line (no header) to outfile:
%   id,n,L,rho,seed,method,seconds,t1,t2,t3,err,err_active,err_nonactive,
%   active_count,groups_active,min_active_width
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','lib'));
maxNumCompThreads(1);
q=load(netfile); A=cellfun(@double,q.A,'UniformOutput',false); c=q.c; truth=q.truth; widths=q.widths;
w=struct('n',200,'L',40,'rho',0.10,'K',3,'M',3,'kappa',10,'seed',17);
[Aw,~,ww]=generate_network(w); Aw=cellfun(@double,Aw,'UniformOutput',false);
fit_method(method,Aw,ww,w); clear Aw
t=zeros(1,3);
for j=1:3
    t0=tic; [pred,active]=fit_method(method,A,widths,c); t(j)=toc(t0);
end
err=NaN; err_act=NaN; err_non=NaN; groups=0; minw=NaN;
if ~isempty(pred), [err,err_act,err_non]=layer_errors(pred,truth,c.M,active); end
if ~isempty(active), groups=numel(unique(truth(active))); minw=min(widths(active)); end
fid=fopen(outfile,'w');
fprintf(fid,'%d,%d,%d,%g,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d\n',c.id,c.n,c.L,c.rho,c.seed, ...
    method,mean(t),t(1),t(2),t(3),err,err_act,err_non,numel(active),groups,minw);
fclose(fid);
end
