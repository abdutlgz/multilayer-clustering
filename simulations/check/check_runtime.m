function check_runtime()
% Small end-to-end run of the Section 4.3 code on a laptop (under a minute):
% one network with n=300, L=60, rho=0.10 is saved and every method is run through
% bench_one.m, as on the cluster but without /usr/bin/time. The real settings
% (n=2000-4000, L=1000-4000) need 2-8 GiB per process and should run on a cluster.
here=fileparts(mfilename('fullpath')); root=fullfile(here,'..');
addpath(fullfile(root,'runtime'),fullfile(root,'lib'));
tmp=fullfile(tempdir,sprintf('check_runtime_%d',feature('getpid'))); mkdir(tmp);
cleanup=onCleanup(@() rmdir(tmp,'s'));
c=struct('id',0,'n',300,'L',60,'rho',0.10,'K',3,'M',3,'kappa',10,'seed',12345);
[A,truth,widths]=generate_network(c); %#ok<ASGLU>
f=fullfile(tmp,'net.mat'); save(f,'A','truth','widths','c','-v7.3');
methods={'Input_only','Algorithm1','All_layer','UASE','Top30','Top100'};
fprintf('\n%-11s %8s %8s %8s\n','method','active','error %','time s');
err=struct();
for k=1:numel(methods)
    out=fullfile(tmp,[methods{k} '.csv']); bench_one(f,methods{k},out);
    x=strsplit(strtrim(fileread(out)),',');
    err.(methods{k})=str2double(x{11});
    fprintf('%-11s %8s %8.1f %8.3f\n',methods{k},x{14},100*str2double(x{11}),str2double(x{7}));
end
assert(err.Top100==err.All_layer,'Top100 should equal all-layer clustering');
fprintf('OK: every method ran; Top100 gives the same error as all-layer clustering.\n');
end
