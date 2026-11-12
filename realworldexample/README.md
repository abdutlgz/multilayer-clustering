# Real-data example: MovieLens (Section 5)

This folder holds the code and results for every figure, table and number in Section 5 of the paper. The results in `results/` are the ones in the paper. You can rebuild the figures and tables from them in a few seconds, check one division on a laptop in about 30 seconds, or rerun everything in about 25 minutes.

## What reproduces what

| Paper | Results | Made by |
|---|---|---|
| Section 5.1: ρ̂, the threshold (254 movies, β⋆ ≈ 0.65), the sizes of the large and small layers | `results/divisions/S032/` | `figures/make_tables.py` |
| Section 5.2: the groups of genres, 244 and 236 of the 256 small layers, 28 of the 32 large layers | `results/divisions/S032/r001.json` | `figures/make_tables.py` |
| Figure 5: groups of Algorithm 1 (M = 2) on the spectral embedding | `results/divisions/S032/r001.json`, `similarity_r001.npz` | `figures/make_figures.py` |
| Figure 6: groups of all layers, three methods, M = 2 and 3 | `results/divisions/S032/r001.json` | `figures/make_figures.py` |
| Section 5.2: the 50 divisions (45 of 50, 95% / 92% of small layers, removal of half of the active layers) | `results/divisions/S032/` | `figures/make_tables.py` |
| Table 3: ARI for 8, 16, 32 and 64 small layers per genre | `results/divisions/S0*/` | `figures/make_tables.py` |
| Figure 7: ARI against the number of small layers per genre | `results/divisions/S0*/` | `figures/make_figures.py` |
| Section 5.2: 19 and 16 active small layers; Algorithm 1 best in 33/46/50/50 (M = 2) and 30/50/50/50 (M = 3) divisions | `results/divisions/S0*/` | `figures/make_tables.py` |

The figures are in `results/figures/` under the file names used in the paper's LaTeX source, and they match the paper's figures pixel for pixel.

## Layout

```
lib/
  data.py            MovieLens: the 5,000 users, the 8 genres, the layers of one random division
  methods.py         the active-layer rule, Algorithm 1, all-layer spectral clustering, the UASE-based method
run/
  run_divisions.py   fits the three methods on 50 random divisions for one number of small layers per genre
figures/
  make_tables.py     prints Table 3 and every number in the text; writes results/summary.csv
  make_figures.py    Figures 5-7
check/
  check_division.py  reruns one division and compares it with results/
results/
  divisions/S008, S016, S032, S064/r001.json ... r050.json   one file per division (see below)
  divisions/S032/similarity_r001.npz                       layer similarity matrix of division 1 (Figure 5)
  summary.csv        one row per division, M and method: ARI, share of small layers in their genre's group, ...
  figures/           Figures 5-7 (PDF and PNG)
get_data.sh          downloads MovieLens 25M and checks its MD5
```

## Requirements

- Python 3.9 or later with numpy, scipy, pandas and matplotlib (`pip install -r requirements.txt`). The paper results were computed with Python 3.9.10, numpy 2.0.2, scipy 1.13.1, pandas 2.3.3 and matplotlib 3.9.4 on a MacBook Pro (Apple M1, 8 cores; numpy uses Apple Accelerate for BLAS).
- About 1.1 GB of memory per process for the runs (measured for 64 small layers per genre, the largest setting). The figures and tables need only the files in `results/`.

## 1. Rebuild the figures and tables from the stored results

```bash
cd realworldexample/figures
python3 make_tables.py       # prints Table 3 and every number in Section 5
python3 make_figures.py      # writes ../results/figures/*.pdf and *.png
```

## 2. Get the data

MovieLens 25M is not in this repository. Download it (about 250 MB) with

```bash
cd realworldexample
./get_data.sh                # into realworldexample/data/ml-25m (git-ignored), MD5 checked
```

or set `ML25M_DIR` to a folder that already has `ratings.csv` and `movies.csv`. The data are from GroupLens, https://grouplens.org/datasets/movielens/25m/ (MD5 of `ml-25m.zip`: `6b51fb2759a8657d3bfcbfc42b592ada`); see their README for the terms of use.

## 3. Check one division (about 30 seconds)

```bash
cd realworldexample/check
python3 check_division.py           # division 1 with 32 small layers per genre (the one in Figures 5-6)
python3 check_division.py 64 50     # any other: number of small layers per genre, division
```

It reruns the division and prints `MATCH` if the active layers and all group labels are identical to `results/`. For 32 small layers per genre it also checks the refit after removing half of the active layers.

## 4. Rerun everything (about 25 minutes)

```bash
cd realworldexample/run
for S in 8 16 32 64; do python3 run_divisions.py $S --workers 6; done
```

New results go to `realworldexample/output/divisions/` (git-ignored), so they never overwrite `results/`. On the 8-core MacBook Pro with 6 workers, the four runs took 2, 2, 7 and 11 minutes. A run that stops can be restarted; finished divisions are skipped. To compare with the paper:

```bash
cd realworldexample/figures
python3 make_tables.py ../output
python3 make_figures.py ../output ../output/figures
```

All results use fixed seeds, so a rerun on the same setup gives labels identical to `results/`. On another BLAS library or processor, the eigenvectors can differ in the last digits, and k-means may then end in a different partition in a borderline division; `check/check_division.py` shows whether this happens on your machine.

## Data and methods

**Layers** (`lib/data.py`):

- Users: the n = 5,000 users with the most ratings (all have at least 40). Rows of every layer are these users.
- Genres: Action, Adventure, Comedy, Crime, Drama, Romance, Sci-Fi and Thriller. Only movies rated by at least one of the 5,000 users are used.
- In each genre, a random 75% of the movies are divided into 4 equal large layers, and the other 25% into S equal small layers (S = 8, 16, 32 or 64; the main setting is S = 32, which gives 288 layers).
- A layer is the binary users × movies matrix: entry (i, j) is 1 if user i rated movie j. The rating values are not used. A movie with several genres appears in layers of each of its genres.
- Division `rep` with S small layers uses the random generator `numpy.random.default_rng([20261108, 7, S, rep])`.

**Active layers** (`lib/methods.py`, `active_rule`): the rule of Section 4. Layer l is active if n_l ≥ log n / ρ̂, that is β_l = log n_l / log n ≥ β⋆, where ρ̂ is the fraction of entries equal to one over all layers. With S = 32, ρ̂ ≈ 0.034 and a layer is active if it has at least 254 movies, which selects exactly the 32 large layers.

**Methods** (`lib/methods.py`, `fit`), all with rank K = 5:

- **Algorithm 1.**
  - Step 1a: U_l are the 5 leading eigenvectors of the hollowed Gram matrix A_l A_lᵀ − diag(A_l A_lᵀ), for every layer.
  - Steps 1b–1c: G(i, j) = ‖U_iᵀ U_j‖²_F over the active layers; k-means on the M leading eigenvectors of G.
  - Steps 2a–2b: for each group, the 5 leading eigenvectors Û^(m) of the hollowed Gram matrix of its active layers put side by side.
  - Step 2c: each non-active layer goes to the group with the largest ‖(Û^(m))ᵀ A_l‖²_F.
- **All-layer spectral clustering:** Steps 1a–1c on all layers.
- **UASE-based method:** V are the 5M leading left singular vectors of [A_1, …, A_L]; each layer gets the feature Vᵀ(A_l A_lᵀ − diag(A_l A_lᵀ))V, scaled to unit norm; spectral clustering of the features (M leading left singular vectors, rows scaled to unit length, k-means).
- k-means: 40 k-means++ starts, the best within-group sum of squares is kept. Eigenvectors: `scipy.sparse.linalg.eigsh` with tolerance 10⁻⁷; the n × n matrices are never formed.

**Seeds:** division `rep` uses k-means seed 8411 + rep; the eigenvectors of layer i use seed 10000 + i, those of group m use 300000 + m, and the UASE common subspace uses 710000.

**Removal** (S = 32 only): in division `rep`, `default_rng([20261109, 77, rep])` removes 2 of the 4 large layers of every genre, and the three methods are refitted on the remaining 272 layers.

**Measures** (`figures/make_tables.py`):

- ARI between the groups of all layers and their genres.
- The share of small layers that are in the same group as most large layers of their genre.

## Format of `results/divisions/S{S}/r{rep}.json`

| Key | Meaning |
|---|---|
| `S`, `rep` | number of small layers per genre, division |
| `genre`, `type` | genre and `large` / `small` of every layer, in layer order (by genre; the 4 large layers first) |
| `ncols`, `nnz` | number of movies and number of ratings of every layer |
| `active`, `rho_hat`, `threshold` | active layers, ρ̂ and the threshold log n / ρ̂ (in movies) |
| `labels["2"]`, `labels["3"]` | group of every layer for M = 2 and 3, for `algorithm1`, `all_layer` and `uase` |
| `removal` (S = 32 only) | `kept` (indices of the remaining layers), their `labels` and `active` flags |

## Where the stored results come from

`results/divisions/` was computed on Nov 11, 2026 with the code in this folder. The paper's numbers were first computed on Nov 9–10, 2026 with earlier scripts; this folder merges them into one code. Before the merge was stored, the merged code was checked against those runs: the group labels are identical in all 200 divisions (S = 8, 16, 32, and the removal refits), and for S = 64, where only the ARIs were kept, the ARIs agree to 10⁻¹⁶. In the earlier runs for S = 32 and 64, a layer was called active if it had at least n log n ratings; for these two settings this selects the same 32 large layers as the rule above, so the results are the same.
