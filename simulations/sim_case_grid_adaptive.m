function case_table = sim_case_grid_adaptive()
% SIM_CASE_GRID_ADAPTIVE
% Returns a curated grid of (n, L, d, rho) values for the corrected ASBM
% simulation study.
%
% Design goal:
%   Smaller networks are paired with larger d values, while larger networks
%   are paired with smaller d values, so the oracle-threshold calibration is
%   learned from nontrivial error regimes instead of near-zero-error cases.
%
% Output:
%   case_table: table with columns
%       n, L, d, rho

c = 0;

case_mat = [
    200   50   0.30
    200   50   0.40
    200   50   0.50
    200   75   0.25
    200   75   0.35
    200   75   0.45
    250   50   0.20
    250   50   0.30
    250   50   0.40
    250   75   0.20
    250   75   0.30
    250  100   0.20
    250  100   0.30
    300   50   0.15
    300   50   0.25
    300   50   0.35
    300   75   0.15
    300   75   0.25
    300  100   0.15
    300  100   0.25
    350   75   0.10
    350   75   0.20
    350   75   0.30
    350  100   0.10
    350  100   0.20
    350  125   0.10
    350  125   0.20
    400  100   0.10
    400  100   0.15
    400  100   0.20
    400  125   0.10
    400  125   0.15
    400  150   0.10
    400  150   0.15
    450  100   0.10
    450  100   0.15
    450  125   0.10
    450  125   0.15
    450  150   0.10
];

rho = (c + case_mat(:,3)) / 2;
case_table = array2table(case_mat, 'VariableNames', {'n','L','d'});
case_table.rho = rho;
end
