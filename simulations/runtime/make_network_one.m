function make_network_one(id, r, netdir)
% Generate and save network r of setting `id` (row of settings.csv). Not timed.
% Seed 8000000 + 1000*id + r.
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','lib'));
S=readmatrix(fullfile(here,'settings.csv')); row=S(S(:,1)==id,:); assert(size(row,1)==1);
if ~isfolder(netdir), mkdir(netdir); end
f=fullfile(netdir,sprintf('s%02d_r%02d.mat',id,r));
if isfile(f), return; end
c=struct('id',id,'n',row(2),'L',row(3),'rho',row(4),'K',3,'M',3,'kappa',10,'seed',8000000+1000*id+r);
[A,truth,widths]=generate_network(c); %#ok<ASGLU>
tmp=[f '.tmp.mat']; save(tmp,'A','truth','widths','c','-v7.3'); movefile(tmp,f);
fprintf('saved %s\n',f);
end
