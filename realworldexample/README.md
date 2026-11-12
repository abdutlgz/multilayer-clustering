# Real-data example: MovieLens (Section 5)

## What this folder shows

We apply the method to real movie ratings from MovieLens 25M, to see whether it finds groups that make sense.

- **Users.** We take the 5,000 users who rated the most movies.
- **Layers.** Each layer is a set of movies of one genre. The layer records which of the 5,000 users rated which of its movies (rated = 1, not rated = 0). The rating values themselves are not used.
- **Large and small layers.** For each of 8 genres (Action, Adventure, Comedy, Crime, Drama, Romance, Sci-Fi, Thriller), a random 75% of the genre's movies are split into 4 **large** layers, and the other 25% into many **small** layers. In the main setting there are 32 small layers per genre, so 8 × (4 + 32) = 288 layers.
- **Active layers.** The large layers have enough data and are *active*; the small ones are not. The rule is the same as in the simulations.

The methods never see the genres. We use the genres only afterwards, to check whether the groups make sense.

**Main result.** With 2 groups, our method (Algorithm 1) puts Action, Adventure, Sci-Fi, Thriller and Crime in one group and Comedy, Drama and Romance in the other, and it places almost every small layer in the group of its genre. The two methods that group all layers at once (all-layer spectral clustering and the UASE-based method) do this much less well, and the gap grows as the small layers get smaller.

Because the split of the movies into layers is random, everything is repeated for **50 random splits** (called *divisions* below), and for **8, 16, 32 and 64 small layers per genre**.

## Where each figure and table comes from

| In the paper | What it shows | Made by |
|---|---|---|
| Figure 5 | the groups found by Algorithm 1 for one division, drawn as a map of the layers | `figures/make_figures.py` |
| Figure 6 | the group of every layer, for all three methods, with 2 and 3 groups | `figures/make_figures.py` |
| Table 3 | how well the groups agree with the genres (ARI), for 8, 16, 32 and 64 small layers per genre | `figures/make_tables.py` |
| Figure 7 | the same as Table 3, as a plot | `figures/make_figures.py` |
| Numbers in the text of Section 5 | for example 244 of 256 small layers, 45 of 50 divisions, and removing half of the active layers | `figures/make_tables.py` |

The figure files are in `results/figures/` under the names used in the paper's LaTeX source, and they are identical to the figures in the paper. `make_tables.py` prints every number, labeled with the part of the section it belongs to.

**ARI** (adjusted Rand index) measures how well two groupings agree: 1 means perfectly, 0 means no better than chance. Here it compares the groups of the layers with their genres. Since there are only 2 or 3 groups but 8 genres, several genres must share a group, so the ARI stays well below 1 even for a sensible grouping.

## Step 1: rebuild the figures and tables (a few seconds)

You need Python 3 with numpy, scipy, pandas and matplotlib (`pip install -r requirements.txt`). You do not need the MovieLens data for this step.

```bash
cd realworldexample/figures
python3 make_figures.py      # writes Figures 5-7 to ../results/figures/
python3 make_tables.py       # prints Table 3 and every number in Section 5
```

## Step 2: download the data (about 250 MB)

```bash
cd realworldexample
./get_data.sh
```

This downloads MovieLens 25M from GroupLens into `realworldexample/data/ml-25m/` and checks that the file is complete. The data are not stored in this repository; see https://grouplens.org/datasets/movielens/25m/ for the terms of use. If you already have the data, skip this step and set `ML25M_DIR` to the folder that contains `ratings.csv` and `movies.csv`.

## Step 3: check one division (about 30 seconds)

```bash
cd realworldexample/check
python3 check_division.py            # division 1 with 32 small layers per genre (the one in Figures 5-6)
python3 check_division.py 64 50      # any other: number of small layers per genre, then the division
```

This reruns all three methods on one division and prints `MATCH` if every result is identical to the stored one.

## Step 4: rerun everything (about 25 minutes)

```bash
cd realworldexample/run
for S in 8 16 32 64; do python3 run_divisions.py $S --workers 6; done
```

`S` is the number of small layers per genre, and each run does 50 divisions. On an 8-core laptop with 6 workers, the four runs took 2, 2, 7 and 11 minutes. New results go to `realworldexample/output/`, so they never overwrite `results/`. If a run stops, start it again: finished divisions are skipped.

To compare the rerun with the paper:

```bash
cd realworldexample/figures
python3 make_tables.py ../output
python3 make_figures.py ../output ../output/figures
```

## What is in each folder

```
lib/
  data.py            reads MovieLens; picks the users and genres; makes the layers of one division
  methods.py         the rule for "active", Algorithm 1, all-layer spectral clustering, the UASE-based method
run/
  run_divisions.py   runs the three methods on 50 divisions, for one number of small layers per genre
figures/
  make_figures.py    Figures 5-7
  make_tables.py     Table 3 and every number in the text; also writes results/summary.csv
check/
  check_division.py  the 30-second check of Step 3
results/
  divisions/S008/ ... S064/            one file per division (r001.json ... r050.json) with the groups found by each method
  divisions/S032/similarity_r001.npz   similarities between the layers of division 1, used for Figure 5
  summary.csv                          one row per division, number of groups and method
  figures/                             Figures 5-7 (PDF and PNG)
get_data.sh          downloads the data (Step 2)
requirements.txt     the Python packages
```

## Technical details

This part is for readers who want the exact settings.

### Requirements

- Python 3.9 or later with numpy, scipy, pandas and matplotlib. The paper results were computed with Python 3.9.10, numpy 2.0.2, scipy 1.13.1, pandas 2.3.3 and matplotlib 3.9.4 on a MacBook Pro (Apple M1, 8 cores; numpy uses Apple Accelerate for BLAS).
- About 1.1 GB of memory per process for the runs (measured for 64 small layers per genre, the largest setting).

### The layers (`lib/data.py`)

- Users: the n = 5,000 users with the most ratings (all have at least 40). Only movies rated by at least one of them are used.
- In each genre, a random 75% of the movies are divided into 4 equal large layers, and the other 25% into S equal small layers.
- A layer is the binary users × movies matrix: entry (i, j) is 1 if user i rated movie j. A movie with several genres appears in layers of each of its genres.
- Division `rep` with S small layers uses the random generator `numpy.random.default_rng([20261108, 7, S, rep])`.

### Active layers (`lib/methods.py`, `active_rule`)

The rule of Section 4: layer l is active if n_l ≥ log n / ρ̂, that is, β_l = log n_l / log n ≥ β⋆. Here n_l is the number of movies in the layer, and ρ̂ is the fraction of entries equal to one over all layers.

- With 32 or 64 small layers per genre: ρ̂ ≈ 0.034, so a layer is active if it has at least 254 movies. This selects exactly the 32 large layers.
- With 8 or 16 small layers per genre, the rule also selects the small layers of the largest genres: 19 small layers (of Comedy, Drama and Thriller) with 8, and the 16 small layers of Drama with 16.

### The methods (`lib/methods.py`, `fit`)

All ranks are K = 5.

- **Algorithm 1.**
  - Step 1a: for every layer, U_l are the 5 leading eigenvectors of the hollowed Gram matrix A_l A_lᵀ − diag(A_l A_lᵀ).
  - Steps 1b–1c: G(i, j) = ‖U_iᵀ U_j‖²_F over the active layers, then k-means on the M leading eigenvectors of G.
  - Steps 2a–2b: for each group, the 5 leading eigenvectors Û^(m) of the hollowed Gram matrix of its active layers placed side by side.
  - Step 2c: each non-active layer goes to the group with the largest ‖(Û^(m))ᵀ A_l‖²_F.
- **All-layer spectral clustering:** Steps 1a–1c on all layers.
- **UASE-based method:** V are the 5M leading left singular vectors of [A_1, …, A_L]. Each layer gets the feature Vᵀ(A_l A_lᵀ − diag(A_l A_lᵀ))V, scaled to unit norm. The features are grouped by spectral clustering: the M leading left singular vectors, rows scaled to unit length, then k-means.
- k-means uses 40 k-means++ starts and keeps the one with the smallest within-group sum of squares. Eigenvectors are computed with `scipy.sparse.linalg.eigsh` (tolerance 10⁻⁷) from matrix–vector products, so the n × n matrices are never formed.

### Random seeds

- Division `rep` uses k-means seed 8411 + rep.
- The eigenvectors of layer i use seed 10000 + i, those of group m use 300000 + m, and the UASE common subspace uses 710000.
- On the same software and machine, a rerun gives exactly the stored results. On another BLAS library or processor, the eigenvectors can differ in the last digits, and k-means may then end in a different grouping in a borderline division; Step 3 shows whether this happens on your machine.

### Removing half of the active layers (32 small layers per genre only)

In division `rep`, `numpy.random.default_rng([20261109, 77, rep])` removes 2 of the 4 large layers of every genre, and the three methods are run again on the remaining 272 layers.

### Measures (`figures/make_tables.py`)

- ARI between the groups of all layers and their genres.
- The share of small layers that are in the same group as most large layers of their genre.

### Format of `results/divisions/S{S}/r{rep}.json`

| Key | Meaning |
|---|---|
| `S`, `rep` | number of small layers per genre, and the division |
| `genre`, `type` | genre and `large` / `small` of every layer, in layer order (by genre; the 4 large layers first) |
| `ncols`, `nnz` | number of movies and number of ratings of every layer |
| `active`, `rho_hat`, `threshold` | active layers, ρ̂, and the threshold log n / ρ̂ (in movies) |
| `labels["2"]`, `labels["3"]` | group of every layer for M = 2 and 3 groups, for `algorithm1`, `all_layer` and `uase` |
| `removal` (S = 32 only) | `kept` (indices of the remaining layers), their `labels` and `active` flags |

### Where the stored results come from

`results/divisions/` was computed on Nov 11, 2026 with the code in this folder. The paper's numbers were first computed on Nov 9–10, 2026 with earlier scripts, which this folder merges into one code. The merged code was checked against those runs: the groups are identical in all 200 divisions (8, 16 and 32 small layers per genre, and the removal runs), and for 64 small layers per genre, where only the ARIs were kept, the ARIs agree to 10⁻¹⁶. The earlier runs with 32 and 64 small layers per genre called a layer active if it had at least n log n ratings; for these two settings this selects the same 32 large layers as the rule above, so the results are the same.
