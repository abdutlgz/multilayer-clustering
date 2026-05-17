# MovieLens Real-Data Experiment

This folder is organized around the MovieLens 25M real-data experiment for the multilayer network clustering paper.

## Run

From the project root in MATLAB:

```matlab
run_movielens_experiment_grid
```

Optional arguments:

```matlab
run_movielens_experiment_grid(overwrite, makeHeatmaps)
```

- `overwrite = false` by default, so existing complete runs are reused.
- `makeHeatmaps = false` by default. Heatmaps are not part of the main real-data outputs.

## Active Files

Project-root runner and helpers:

- `run_movielens_experiment_grid.m`
- `make_movielens_realdata_tables.m`
- `make_movielens_realdata_figures.m`
- `get_movielens_final_labels.m`
- `movielens_cluster_composition.m`
- `compute_movielens_theta_full.m`
- `write_latex_table_simple.m`
- `write_movielens_outcome_summary.m`

Required MovieLens helpers in `ml-25m/`:

- `load_movielens25_tables.m`
- `build_movielens_genre_splits.m`
- `classify_inactive_layers_genre.m`

Raw MovieLens files stay local in `ml-25m/` and should not be uploaded:

- `ratings.csv`
- `movies.csv`

## Outputs

The runner creates or updates:

- `results_movielens_builds/`: saved layer constructions for uneven/equal/positive-only builds.
- `results_movielens/`: saved pipeline runs for the beta/M grid.
- `tables_movielens/`: CSV, LaTeX tables, and `movielens_experiment_outcomes.txt`.
- `figures_movielens/`: beta-width plots, active/non-active embeddings, cluster composition plots, and sensitivity plots.
- `paper_movielens/`: curated paper-facing subset of the most useful figures and tables.

You can regenerate only the curated paper package after the grid has run:

```matlab
make_movielens_paper_package
```

The focused paper story is:

1. Main result: `uneven_all`, `beta_star = 0.75`, `M = 3`.
2. Coarse comparison: `uneven_all`, `beta_star = 0.75`, `M = 2`.
3. Over-segmentation diagnostic: `uneven_all`, `beta_star = 0.75`, `M = 4`.
4. Robustness check: `equal_all`, `beta_star = 0.75`, `M = 3`.
5. Signal-change check: `uneven_positive4`, `beta_star = 0.80`, `M = 3`.

## Synthetic Validation Simulations

To generate controlled evidence for the method, run:

```matlab
run_method_validation_simulations
```

This creates `simulations_method_validation/` with:

- M=2 and M=3 representative embedding figures.
- Error-vs-threshold curves across layer-width heterogeneity regimes.
- Signal-sensitivity curves comparing the proposed split pipeline, all-layer clustering, and an oracle split classifier.
- Inactive-layer classification margin curves.
- CSV and LaTeX summary tables.

Use a smaller or larger number of Monte Carlo repetitions with:

```matlab
run_method_validation_simulations(true, 10)
run_method_validation_simulations(true, 50)
```

## Real-Data Scenario Experiments

To test whether the MovieLens embedding story persists across real
construction choices, run:

```matlab
run_movielens_realdata_scenarios
```

This creates `movielens_scenario_experiments/` with M=2 and M=3 embeddings
for scenarios that vary:

- number of selected users: 3000, 8000, 12000
- number of genre splits/layers: 4, 8, 12 splits per genre
- genre universe: 8 core genres vs 12 broader genres
- retained-user activity threshold: 40 vs 80 ratings
- all ratings vs positive-only ratings
- uneven vs equal layer widths

Each scenario chooses an adaptive `beta_star` that keeps roughly half the
layers active and half non-active, so the comparison is about the real-data
construction rather than arbitrary beta-grid points.

After the scenario runs finish, compare against same-input baselines with:

```matlab
run_movielens_scenario_algorithm_comparison
```

This compares the proposed split/subspace classifier to:

- all-layer spectral clustering
- split active clustering with nearest-centroid embedding classification
- k-means using only layer size/density covariates

The point is to check whether the proposed pipeline gives interpretable
genre-family structure beyond ordinary all-layer clustering or simple
width/density effects.

## Split-Advantage Stress Experiments

The broad real-data scenarios above showed that all-layer spectral clustering
can look as good as, or better than, the split pipeline when most layers carry
usable signal. To test the regime where the split method should help, run:

```matlab
run_movielens_split_advantage_scenarios
```

This creates `movielens_split_advantage_experiments/`. Each scenario builds a
few wide/high-signal anchor layers per genre and many thin, edge-subsampled
weak layers. The proposed method clusters only the anchor layers, then asks
whether the weak layers classify back to the same genre-family clusters.

Use:

```matlab
run_movielens_split_advantage_scenarios(true)
```

to rebuild everything from scratch.

Main outputs:

- `tables/split_advantage_algorithm_summary.csv`
- `tables/split_advantage_readme.txt`
- `figures/split_advantage_metrics_M2.pdf`
- `figures/split_advantage_metrics_M3.pdf`
- `figures/stress_algorithm_embedding_<scenario>_M<M>.pdf`

The main score is `Weak_anchor_agreement`. This asks whether weak/noisy layers
from a genre land in the same cluster as that genre's strong anchor layers.

## Legacy Code

Unused simulation and older MovieLens scripts were moved to:

- `legacy_unused_code/simulation_code/`
- `legacy_unused_code/ml25m_old_scripts/`
- `legacy_unused_code/Old Code/`
- `legacy_unused_code/Secondary/`
- `legacy_unused_code/organized_simulations/`
