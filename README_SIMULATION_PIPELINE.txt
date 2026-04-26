Clean simulation pipeline for the corrected ASBM generator
==========================================================

Main idea
---------
This folder replaces the older mixed batch scripts with a clean
three-step workflow:

1. STEP1_single_case_beta_sweep.m
   Pick one (n, L, d), sweep the beta threshold, and verify the U-shaped
   error curve with an oracle minimum.

2. STEP2_build_oracle_delta_table.m
   Run many selected (n, L, d) cases, compute the oracle threshold in each
   case, and save the clean oracle table:
       n, L, d, rho, errMin, betaOracle, deltaOracle

3. STEP3_fit_delta_rule_and_compare.m
   Fit deltaOracle as a regression in log(n), log(L), and log(d), then
   compare:
       oracle threshold
       empirical threshold from deltaHat
       baseline clustering without thresholding

Files
-----
sim_case_grid_adaptive.m
   Curated list of (n, L, d) configurations. Smaller networks are paired
   with larger d values, and larger networks with smaller d values.

sim_oracle_beta_curve.m
   Shared helper that computes the full error-vs-threshold curve for one
   configuration.

sim_split_pipeline_error.m
   Shared helper for the split-and-classify total error.

sim_baseline_error.m
   Shared helper for the baseline with no active/non-active split.

Outputs
-------
All new outputs are written inside:
   organized_simulations/results/

Recommended run order in MATLAB
-------------------------------
1. Run STEP1_single_case_beta_sweep.m
2. Run STEP2_build_oracle_delta_table.m
3. Run STEP3_fit_delta_rule_and_compare.m

Notes
-----
- The corrected data generator is ASBM_varNL_layerB.m
- ASBM_varNL.m is now only a compatibility wrapper
- The old batch files are left untouched for reference
