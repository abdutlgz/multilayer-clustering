function s=select_active_layers(A,widths,n)
% Data-driven threshold of the paper, eq. (sim:threshold):
%   layer l is active if n_l >= log(n)/rho_hat,
% where rho_hat is the fraction of ones over all adjacency matrices.
% Equivalently beta_hat = [log(1/rho_hat) + log log n]/log n. There is no
% fallback: if no layer passes, the active set is empty.
L=numel(widths); ones_total=sum(cellfun(@nnz,A)); entries=n*sum(widths); rho=ones_total/entries;
raw=log(n)/rho; active=find(widths(:)'>=raw);
if isempty(active), be=NaN; else, be=log(min(widths(active)))/log(n); end
s=struct('rho_hat',rho,'raw_width',raw,'beta_formula',log(raw)/log(n),'beta_effective',be, ...
    'active_ids',active,'active_count',numel(active),'L',L);
end
