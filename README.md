# Multilayer Network Clustering Experiments

This repository contains MATLAB code for experiments in a multilayer network
clustering and inactive-layer classification project. The main real-data example
uses MovieLens 25M to build genre-conditioned user--movie bipartite layers.

The goal is not movie recommendation or rating prediction. The goal is to test
whether an active-layer clustering pipeline can recover interpretable
between-layer structure and then classify weaker non-active layers.

## Repository Structure

```text
realworldexample/
  MovieLens real-data runners, helpers, paper summary, and selected outputs.

simulations/
  Controlled simulation pipeline for validating the method under known signal
  and layer-width regimes.
```

The most important documentation files are:

```text
realworldexample/SUMMARY.md
realworldexample/README_movielens_experiment.md
simulations/README_SIMULATION_PIPELINE.txt
```

Read `realworldexample/SUMMARY.md` first if you want the paper-facing story:
what was tested, which cases worked best, and which figures/tables are most
useful.

## Main Real-Data Experiment

The real-data example uses MovieLens 25M. Each layer is a rectangular
user--movie bipartite adjacency matrix:

```text
A^{(l)} in R^{n x n_l}
```

Rows are a common set of users. Columns are movie subsets created from genre
splits. The default edge is binary:

```text
A(i,j) = 1 if user i rated movie j
```

The main MovieLens scripts are in `realworldexample/`.

### Broad Scenario Experiments

Run:

```matlab
run_movielens_realdata_scenarios
```

This tests real construction choices including:

- number of users: 3,000; 8,000; 12,000
- number of layers: 32, 64, 96
- 8 core genres vs a broader 12-genre universe
- all ratings vs positive-only ratings
- uneven vs equal-width layer splits

These runs are used first in the paper story to show that the method recovers
interpretable M=2 and M=3 genre-family structure in real MovieLens layers.

### Algorithm Comparison

Run:

```matlab
run_movielens_scenario_algorithm_comparison
```

This compares the proposed split/subspace classifier with:

- all-layer spectral clustering
- split active clustering with nearest-centroid embedding classification
- k-means using only layer size/density covariates

### Split-Advantage Stress Experiments

Run:

```matlab
run_movielens_split_advantage_scenarios
```

These scenarios create the regime where the active/inactive split should help:
a few wide/high-signal anchor layers per genre and many thin, edge-subsampled
weak layers. The key metric is `Weak_anchor_agreement`, which checks whether
weak layers classify back to the same cluster as their genre's strong anchor
layers.

Best paper-facing cases:

```text
Broad M=2:       users3000_core8_s8
Broad M=3:       embedding_paircoords_users3000_core8_s8_M3
Stress M=2:      stress_u5000_anchor4_weak16_keep20
Stress M=3:      stress_u5000_anchor4_weak16_keep20
Hard stress:     stress_u5000_anchor4_weak24_keep10
Best baseline:   proposed split/subspace vs all-layer spectral clustering
```

## Simulation Experiments

The controlled simulations are in `simulations/`. They test the method under
known signal and layer-width regimes.

Typical workflow:

```matlab
STEP1_single_case_beta_sweep
STEP2_build_oracle_delta_table
STEP3_fit_delta_rule_and_compare
STEP4_make_paper_figures
```

See:

```text
simulations/README_SIMULATION_PIPELINE.txt
```

for the detailed simulation pipeline.

## Data Policy

Raw MovieLens files are not included in this repository.

To reproduce the real-data runs, download MovieLens 25M locally and place:

```text
ratings.csv
movies.csv
```

inside:

```text
realworldexample/ml-25m/
```

Do not commit raw MovieLens data, generated `.mat` files, or large local result
folders.

## Recommended Reading Order

1. `realworldexample/SUMMARY.md`
2. `realworldexample/README_movielens_experiment.md`
3. `simulations/README_SIMULATION_PIPELINE.txt`

