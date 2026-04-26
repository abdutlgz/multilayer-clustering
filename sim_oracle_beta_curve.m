function out = sim_oracle_beta_curve(n, L, d, opts)
% SIM_ORACLE_BETA_CURVE
% For one (n, L, d) configuration, sweep the threshold beta_star, run the
% corrected ASBM simulation, and estimate the oracle threshold that
% minimizes the total split-and-classify error.
%
% Required inputs:
%   n, L, d
%
% opts fields:
%   M, K, c, w, Nruns, beta_min, beta_max, beta_star_list
%
% Output struct fields:
%   n, L, d, rho
%   nL_vec, beta_vec
%   beta_star_list, err_curve, active_count
%   err_min, beta_oracle, delta_oracle

[nL_vec, ~] = gen_nl_beta(n, L, opts.beta_min, opts.beta_max);
nL_vec = max(nL_vec, opts.K);
beta_vec = log(nL_vec) ./ log(n);

beta_star_list = opts.beta_star_list(:)';
err_curve = zeros(size(beta_star_list));
active_count = zeros(size(beta_star_list));

for ib = 1:numel(beta_star_list)
    beta_star = beta_star_list(ib);
    idx_active = find(beta_vec >= beta_star);
    idx_nonactive = find(beta_vec < beta_star);
    active_count(ib) = numel(idx_active);

    errs = zeros(1, opts.Nruns);
    for r = 1:opts.Nruns
        rng(r);
        [A, ~, ~, ~, ~, label] = ASBM_varNL_layerB( ...
            n, nL_vec, opts.K, L, opts.M, opts.c, d, opts.w);
        errs(r) = sim_split_pipeline_error( ...
            A, label, idx_active, idx_nonactive, opts.M, opts.K, n, L);
    end
    err_curve(ib) = mean(errs);
end

[err_min, jbest] = min(err_curve);
beta_oracle = beta_star_list(jbest);
rho = (opts.c + d) / 2;
delta_oracle = (beta_oracle * log(n) + log(rho)) / log(log(n)) - 1;

out = struct();
out.n = n;
out.L = L;
out.d = d;
out.rho = rho;
out.nL_vec = nL_vec;
out.beta_vec = beta_vec;
out.beta_star_list = beta_star_list;
out.err_curve = err_curve;
out.active_count = active_count;
out.err_min = err_min;
out.beta_oracle = beta_oracle;
out.delta_oracle = delta_oracle;
end
