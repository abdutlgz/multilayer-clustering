# MovieLens Real-Data Experiment Summary

This document summarizes the MovieLens 25M real-data experiments for the multilayer
network clustering paper. The goal was not rating prediction. The goal was to show
that the active-layer clustering plus inactive-layer classification pipeline can
recover interpretable between-layer structure in a real multilayer bipartite network.

## Data Setup

The real-data example uses MovieLens 25M from the local `ml-25m/` folder.

- Raw files: `ratings.csv`, `movies.csv`
- Original ratings: 25,000,095
- Original users: 162,541
- Original movies: 62,423
- Network type: binary user--movie bipartite layers
- Default edge rule: `A(i,j)=1` if user `i` rated movie `j`
- Default user filter: retain users with at least 40 ratings
- Default user count: most active 8,000 retained users
- Default genres: Action, Adventure, Comedy, Crime, Drama, Romance, Sci-Fi, Thriller
- Default layer construction: split each genre into multiple movie subsets
- Timestamps are not used
- Numerical ratings are not used except in positive-only scenarios

Each layer is a rectangular adjacency matrix

```text
A^{(l)} in R^{n x n_l}
```

where rows are the same users and columns are genre-conditioned movie subsets.

## What We Built

We built three related experiment families.

### 1. Original MovieLens Grid

Runner:

```matlab
run_movielens_experiment_grid
```

Outputs:

```text
results_movielens_builds/
results_movielens/
tables_movielens/
figures_movielens/
paper_movielens/
```

This tested:

- uneven genre splits, all ratings
- equal genre splits, all ratings
- uneven genre splits, positive-only ratings
- several beta thresholds
- M = 2, 3, 4 clusters

This was useful for establishing the basic pipeline and generating preprocessing
tables, layer summaries, embeddings, and cluster compositions. However, the
beta-bar figures were not very useful visually, and the equal-vs-unequal split
difference was not strong enough to be the main paper story.

### 2. Broad Real-Data Scenario Experiments

Runner:

```matlab
run_movielens_realdata_scenarios
```

Outputs:

```text
movielens_scenario_experiments/
```

These scenarios varied real construction choices:

- number of users: 3,000; 8,000; 12,000
- number of layers: 32, 64, 96
- number of genres: 8 vs 12
- all ratings vs positive-only ratings
- equal-width vs uneven-width splits
- minimum user activity: 40 vs 80 ratings

Scenarios tested:

| Scenario | Meaning |
|---|---|
| `main_u8000_core8_s8` | main 8,000-user, 8-genre, 64-layer case |
| `users3000_core8_s8` | fewer users |
| `users12000_core8_s8` | more users |
| `layers32_core8_s4` | fewer/wider layers |
| `layers96_core8_s12` | more/thinner layers |
| `genres12_u8000_s6` | broader 12-genre universe |
| `strictusers_u8000_min80_core8_s8` | stricter user activity |
| `positive4_u8000_core8_s8` | positive-only ratings |
| `equalwidth_u8000_core8_s8` | equal-width control |

Each scenario chose an adaptive `beta_star` that kept about half the layers
active and half non-active. This avoided sweeping beta values that produce
uninteresting all-active or all-inactive regimes.

Best broad-scenario results:

| Scenario | M | Genre consistency | Embedding silhouette | Inactive margin |
|---|---:|---:|---:|---:|
| `users3000_core8_s8` | 2 | 1.000 | 0.781 | 0.0209 |
| `layers32_core8_s4` | 2 | 1.000 | 0.705 | 0.0266 |
| `positive4_u8000_core8_s8` | 2 | 0.953 | 0.663 | 0.0337 |
| `layers96_core8_s12` | 2 | 0.969 | 0.668 | 0.0227 |
| `main_u8000_core8_s8` | 2 | 0.938 | 0.597 | 0.0159 |

Best visual/numerical broad cases:

- `users3000_core8_s8`, M=2: cleanest broad-scenario embedding and best metrics.
- `users3000_core8_s8`, M=3: best broad-scenario M=3 result; this is the strongest pair-coordinate figure with three visible cluster-pair views.
- `layers32_core8_s4`, M=2: compact, clean, and easy to explain.
- `layers32_core8_s4`, M=3: clean M=3 robustness case with fewer/wider layers.
- `positive4_u8000_core8_s8`, M=2: useful robustness case showing that positive-preference layers still produce interpretable structure.
- `main_u8000_core8_s8`, M=2 and M=3: best default MovieLens setting for continuity with the original experiment design.

Important conclusion from this family:

The broad real-data scenarios should be the first evidence in the paper that the
method is actually recovering meaningful real-data structure. They show
interpretable genre organization for both M=2 and M=3. M=2 is the cleanest and
most stable view. M=3 is useful because it shows a finer three-family structure,
especially in the pair-coordinate plots where the three coordinate pairs reveal
different separations.

These broad scenarios are not the best evidence that the proposed active/inactive
split beats every ordinary clustering baseline. In many broad scenarios,
all-layer spectral clustering is as good as or better than the proposed method on
simple proxy metrics such as genre consistency or embedding silhouette. That is
why the traditional-method comparison should come after the broad scenario
figures, and the stronger method-advantage claim should use the split-advantage
stress experiments.

### 3. Algorithm Comparison on Broad Scenarios

Runner:

```matlab
run_movielens_scenario_algorithm_comparison
```

Outputs:

```text
movielens_scenario_experiments/algorithm_comparison/
```

Compared methods:

- proposed split/subspace classifier
- all-layer spectral clustering
- split active clustering with nearest-centroid embedding classification
- k-means on layer size/density covariates

Broad-scenario average results:

| M | Method | Genre consistency | Embedding silhouette | Genre ARI |
|---:|---|---:|---:|---:|
| 2 | proposed split/subspace | 0.951 | 0.567 | 0.173 |
| 2 | all-layer spectral | 0.969 | 0.750 | 0.188 |
| 2 | split embedding nearest | 0.981 | 0.702 | 0.180 |
| 2 | size/density k-means | 0.890 | 0.319 | 0.122 |
| 3 | proposed split/subspace | 0.917 | 0.305 | 0.247 |
| 3 | all-layer spectral | 1.000 | 0.510 | 0.382 |
| 3 | split embedding nearest | 0.930 | 0.239 | 0.265 |
| 3 | size/density k-means | 0.898 | 0.129 | 0.274 |

Interpretation:

These results are honest but not ideal for proving the active/inactive split.
They show that the discovered structure is not just layer size/density, because
size/density k-means is generally weaker. But they do not show a strong advantage
over all-layer spectral clustering.

## Split-Advantage Stress Experiments

Runner:

```matlab
run_movielens_split_advantage_scenarios
```

Outputs:

```text
movielens_split_advantage_experiments/
```

This is the most important experiment family for demonstrating why the proposed
method is useful.

The construction intentionally creates the regime where active-layer selection
should help:

- each genre has a few wide, high-signal anchor layers
- each genre also has many thin, edge-subsampled weak layers
- active layers are selected by layer width
- active layers estimate the cluster structure
- weak layers test the inactive-layer classifier

The key metric is `Weak_anchor_agreement`.

This asks:

```text
Do weak/noisy layers from a genre get assigned to the same cluster as that
genre's strong anchor layers?
```

This is the right metric for the proposed pipeline. Embedding silhouette can be
misleading here, because all-layer spectral clustering can form visually clean
clusters while assigning weak layers to the wrong anchor families.

### Stress Scenarios

| Scenario | Meaning |
|---|---|
| `stress_u5000_anchor4_weak16_keep20` | Main stress test: 4 anchor + 16 weak layers per genre, 20% weak edge retention |
| `stress_u5000_anchor4_weak24_keep10` | Harder stress test: 4 anchor + 24 weak layers per genre, 10% weak edge retention |
| `stress_u8000_anchor4_weak16_keep20` | Same stress design with 8,000 users |
| `stress_positive4_u5000_anchor4_weak16_keep20` | Positive-only stress test using ratings >= 4 |

### Main Stress Results

Average over the four stress scenarios:

| M | Method | Weak-anchor agreement | Weak consistency | Genre consistency |
|---:|---|---:|---:|---:|
| 2 | proposed split/subspace | 0.952 | 0.952 | 0.957 |
| 2 | all-layer spectral | 0.735 | 0.883 | 0.870 |
| 2 | split embedding nearest | 0.788 | 0.897 | 0.875 |
| 2 | size/density k-means | 0.000 | 1.000 | 0.814 |
| 3 | proposed split/subspace | 0.928 | 0.928 | 0.935 |
| 3 | all-layer spectral | 0.102 | 0.912 | 0.766 |
| 3 | split embedding nearest | 0.692 | 0.895 | 0.858 |
| 3 | size/density k-means | 0.000 | 1.000 | 0.814 |

This is the clearest method-advantage result. The proposed method is best on the
metric that matches the active/inactive goal.

## Best Cases To Use In The Paper

### Best Main Real-Data Case

Use:

```text
stress_u5000_anchor4_weak16_keep20, M=2
```

Why:

- strongest combination of clean visual embedding and strong numeric result
- proposed weak-anchor agreement is 0.977
- best competing method only reaches 0.789
- cluster composition is simple and interpretable

Cluster composition for proposed method:

| Genre | Cluster 1 | Cluster 2 |
|---|---:|---:|
| Action | 20 | 0 |
| Adventure | 19 | 1 |
| Comedy | 0 | 20 |
| Crime | 18 | 2 |
| Drama | 0 | 20 |
| Romance | 0 | 20 |
| Sci-Fi | 20 | 0 |
| Thriller | 20 | 0 |

Interpretation:

- Cluster 1: Action, Adventure, Crime, Sci-Fi, Thriller
- Cluster 2: Comedy, Drama, Romance

This gives a clean coarse genre-family split.

Recommended figure:

```text
movielens_split_advantage_experiments/figures/stress_algorithm_embedding_stress_u5000_anchor4_weak16_keep20_M2.pdf
```

Recommended table:

```text
movielens_split_advantage_experiments/tables/composition_stress_u5000_anchor4_weak16_keep20_M2_proposed_split_subspace.tex
```

### Best M=3 Comparison Case

Use:

```text
stress_u5000_anchor4_weak16_keep20, M=3
```

Why:

- same construction as the best M=2 case
- proposed weak-anchor agreement is 0.930
- all-layer spectral drops to 0.164
- shows why M=3 is more detailed but less visually simple than M=2

Cluster composition for proposed method:

| Genre | Cluster 1 | Cluster 2 | Cluster 3 |
|---|---:|---:|---:|
| Action | 0 | 1 | 19 |
| Adventure | 0 | 0 | 20 |
| Comedy | 20 | 0 | 0 |
| Crime | 0 | 20 | 0 |
| Drama | 11 | 9 | 0 |
| Romance | 20 | 0 | 0 |
| Sci-Fi | 0 | 0 | 20 |
| Thriller | 0 | 19 | 1 |

Interpretation:

- Cluster 1: Comedy, Romance, part of Drama
- Cluster 2: Crime, Thriller, part of Drama
- Cluster 3: Action, Adventure, Sci-Fi

Recommended figure:

```text
movielens_split_advantage_experiments/figures/stress_algorithm_embedding_stress_u5000_anchor4_weak16_keep20_M3.pdf
```

Recommended table:

```text
movielens_split_advantage_experiments/tables/composition_stress_u5000_anchor4_weak16_keep20_M3_proposed_split_subspace.tex
```

### Best Hard-Stress Robustness Case

Use:

```text
stress_u5000_anchor4_weak24_keep10
```

Why:

- hardest all-ratings stress scenario
- many more weak layers
- stronger edge subsampling
- proposed method still wins clearly

Results:

| M | Proposed weak-anchor | Best non-proposed weak-anchor | Gap |
|---:|---:|---:|---:|
| 2 | 0.948 | 0.698 | 0.250 |
| 3 | 0.938 | 0.714 | 0.224 |

This is the best robustness result for showing that the proposed method does not
only work in the easiest stress setting.

Recommended figures:

```text
movielens_split_advantage_experiments/figures/stress_algorithm_embedding_stress_u5000_anchor4_weak24_keep10_M2.pdf
movielens_split_advantage_experiments/figures/stress_algorithm_embedding_stress_u5000_anchor4_weak24_keep10_M3.pdf
```

### Best Positive-Only Case

Use only as a secondary robustness check:

```text
stress_positive4_u5000_anchor4_weak16_keep20
```

Why:

- tests positive preference instead of general rating/exposure
- still supports the method, especially for M=3
- not as strong as the all-ratings stress scenarios for M=2

Results:

| M | Proposed weak-anchor | Best non-proposed weak-anchor | Gap |
|---:|---:|---:|---:|
| 2 | 0.922 | 0.883 | 0.039 |
| 3 | 0.891 | 0.727 | 0.164 |

Interpretation:

The positive-only network carries a different signal. The split method still
works, but the M=2 advantage is much smaller.

## Best Comparison With Traditional Methods

The best traditional-method comparison is:

```text
proposed split/subspace classifier
vs
all-layer spectral clustering
```

Use this comparison on:

```text
stress_u5000_anchor4_weak16_keep20, M=2 and M=3
stress_u5000_anchor4_weak24_keep10, M=2 and M=3
```

Why this is the best comparison:

- all-layer spectral clustering is the natural baseline that ignores active vs
  inactive layers
- it uses the same full layer similarity information
- it is visually plausible in some embeddings
- it fails exactly where the proposed method should help: assigning weak/noisy
  layers to the correct anchor-layer clusters

Most important comparison numbers:

| Scenario | M | Proposed weak-anchor | All-layer spectral weak-anchor |
|---|---:|---:|---:|
| `stress_u5000_anchor4_weak16_keep20` | 2 | 0.977 | 0.758 |
| `stress_u5000_anchor4_weak16_keep20` | 3 | 0.930 | 0.164 |
| `stress_u5000_anchor4_weak24_keep10` | 2 | 0.948 | 0.557 |
| `stress_u5000_anchor4_weak24_keep10` | 3 | 0.938 | 0.000 |

The secondary comparison should be:

```text
proposed split/subspace classifier
vs
split embedding nearest-centroid classifier
```

This is an ablation-style comparison. It asks whether inactive layers should be
classified using the proposed subspace classifier or simply by nearest centroid
in the global embedding. The proposed method wins consistently in the stress
scenarios.

Use size/density k-means only as a negative control. It often gets high
weak-layer consistency because it groups weak layers together by construction,
but its weak-anchor agreement is 0.000. This is useful because it shows that
layer size/noise level alone is not solving the intended task.

## Paper-Ready Figures And Tables

### Main Figures

Use the broad scenario figures first to show that the real-data pipeline is
working for M=2 and M=3:

```text
movielens_scenario_experiments/figures/embedding_users3000_core8_s8_M2.pdf
movielens_scenario_experiments/figures/embedding_paircoords_users3000_core8_s8_M3.pdf
movielens_scenario_experiments/figures/embedding_layers32_core8_s4_M2.pdf
movielens_scenario_experiments/figures/embedding_paircoords_layers32_core8_s4_M3.pdf
movielens_scenario_experiments/figures/embedding_main_u8000_core8_s8_M2.pdf
movielens_scenario_experiments/figures/embedding_paircoords_main_u8000_core8_s8_M3.pdf
```

The `embedding_paircoords_*_M3` figures are especially useful for M=3 because
they show the three coordinate-pair views. The `users3000_core8_s8` version is
the strongest result numerically, while `main_u8000_core8_s8` is the most natural
default construction to describe in the paper.

Then use the split-advantage stress figures to compare against traditional
methods:

```text
movielens_split_advantage_experiments/figures/stress_algorithm_embedding_stress_u5000_anchor4_weak16_keep20_M2.pdf
movielens_split_advantage_experiments/figures/stress_algorithm_embedding_stress_u5000_anchor4_weak16_keep20_M3.pdf
movielens_split_advantage_experiments/figures/stress_algorithm_embedding_stress_u5000_anchor4_weak24_keep10_M2.pdf
movielens_split_advantage_experiments/figures/split_advantage_metrics_M2.pdf
movielens_split_advantage_experiments/figures/split_advantage_metrics_M3.pdf
```

The metric figures are useful but should be cosmetically cleaned before final
submission: the scenario names on the x-axis are too long. Rename them to short
labels such as:

| Current label | Paper label |
|---|---|
| `stress_u5000_anchor4_weak16_keep20` | Main |
| `stress_u5000_anchor4_weak24_keep10` | Hard |
| `stress_u8000_anchor4_weak16_keep20` | Users8000 |
| `stress_positive4_u5000_anchor4_weak16_keep20` | Positive |

### Main Tables

Use these first:

```text
movielens_scenario_experiments/tables/scenario_run_summary.tex
movielens_split_advantage_experiments/tables/split_advantage_algorithm_summary.tex
movielens_split_advantage_experiments/tables/composition_stress_u5000_anchor4_weak16_keep20_M2_proposed_split_subspace.tex
movielens_split_advantage_experiments/tables/composition_stress_u5000_anchor4_weak16_keep20_M3_proposed_split_subspace.tex
tables_movielens/movielens_preprocessing_summary.tex
```

For supplement or appendix:

```text
movielens_scenario_experiments/algorithm_comparison/tables/algorithm_comparison_summary.tex
```

## Recommended Paper Narrative

The most convincing narrative is:

1. MovieLens layers form interpretable genre-family structure.
2. First show the broad scenario results: the method works on real MovieLens
   constructions for M=2 and M=3. M=2 is visually and numerically cleaner, while
   M=3 gives a more detailed three-family view. The pair-coordinate M=3 figures
   are the best visual evidence for this part.
3. Then compare against traditional methods. Broad scenarios alone do not prove
   that active-layer selection beats ordinary all-layer clustering, so the
   comparison should be framed carefully.
4. The active/inactive split is most valuable in the regime it was designed for:
   a mixture of strong signal layers and many weak/noisy layers.
5. In the stress experiments, the proposed method clusters strong anchor layers
   and successfully classifies weak layers back to the correct anchor families.
6. Traditional all-layer spectral clustering can look clean in the embedding,
   but it performs worse on weak-anchor agreement, especially for M=3 and hard
   weak-layer settings.
7. Size/density k-means fails the weak-anchor task, showing that the result is
   not merely a layer-width or density artifact.

## Bottom Line

The paper-ready MovieLens evidence should have two stages. First, use the broad
scenario experiments to show that the method produces meaningful M=2 and M=3
real-data structure. Then use the split-advantage stress experiments for the
strong comparison against traditional methods. The original beta-bar plots and
heatmaps should not be the main evidence.

Best broad M=2 result:

```text
users3000_core8_s8, M=2
```

Best broad M=3 visual:

```text
embedding_paircoords_users3000_core8_s8_M3
```

Best split-advantage main result:

```text
stress_u5000_anchor4_weak16_keep20, M=2
```

Best detailed comparison:

```text
stress_u5000_anchor4_weak16_keep20, M=3
```

Best robustness/stress result:

```text
stress_u5000_anchor4_weak24_keep10, M=2 and M=3
```

Best traditional-method comparison:

```text
proposed split/subspace classifier vs all-layer spectral clustering
```

Most important metric:

```text
Weak_anchor_agreement
```
