# Simulation study (Section 4)

This folder holds the code and results for every table and figure in Section 4 of the paper. The results in `results/` are the ones in the paper. You can rebuild the figures and tables from them in a few seconds, or rerun the simulations on a cluster.

## What reproduces what

| Paper | Results file | Made by |
|---|---|---|
| Figure 1: error against a fixed threshold β⋆ | `results/accuracy/curves.csv`, `settings_summary.csv` | `figures/make_figures.py` |
| Table 1: active layers, clustering and classification error | `results/accuracy/settings_summary.csv` | `figures/make_tables.py` |
| Figure 2: Algorithm 1, all-layer clustering and the UASE-based method against n | `results/accuracy/settings_summary.csv` | `figures/make_figures.py` |
| Table 2: error, time and memory for three settings | `results/runtime/summary.csv` | `figures/make_tables.py` |
| Figure 3: error, time and memory in settings 1–8 | `results/runtime/summary.csv` | `figures/make_figures.py` |
| Figure 4: sweep over the number of active layers | `results/active_sweep/runs.csv` | `figures/make_figures.py` |

The figures are in `results/figures/`, and they match the PDFs in the paper pixel for pixel.

## Layout

```
lib/            shared code
  generate_network.m      simulated network (Section 4.1)
  select_active_layers.m  the threshold n_l >= log n / rho_hat
  fit_method.m            Algorithm 1, all-layer, UASE-based, Algorithm 1 with the XX% widest layers active
  leading_space.m         leading singular vectors (eigs or irlba)
  layer_errors.m          total, clustering and classification error
  irlba.m                 third-party, J. Baglama and L. Reichel (see the file header)
accuracy/       Section 4.2: 208 settings x 100 networks (Figures 1-2, Table 1)
runtime/        Section 4.3: 9 settings x 10 networks (Table 2, Figure 3) and the sweep (Figure 4)
figures/        make_figures.py, make_tables.py
check/          short checks that run on a laptop
results/        results used in the paper (CSV) and the figures
```

## Requirements

- MATLAB with the Statistics and Machine Learning Toolbox. The paper results were computed with R2023a on the UCF Stokes cluster (Slurm), using Intel Xeon Gold 6240R (Cascade Lake) nodes for the timing runs.
- Python 3 with pandas, numpy and matplotlib for the figures (tested with Python 3.9, pandas 2.3, matplotlib 3.9). `runtime/summarize.py` needs only the standard library.
- Linux with GNU `/usr/bin/time` and `lscpu`, for the memory measurements in `runtime/`.

## 1. Rebuild figures and tables from the stored results

```bash
cd simulations/figures
python3 make_figures.py        # writes ../results/figures/*.pdf and *.png
python3 make_tables.py         # prints Table 1, Table 2 and the numbers of Figures 3-4
```

## 2. Check the code on a laptop

In MATLAB, from `simulations/check`:

```matlab
check_accuracy          % reruns 10 networks of one Section 4.2 setting and compares with results/accuracy (~15 s)
check_accuracy(72,100)  % all 100 networks of that setting, including its fixed-threshold curve (~1.5 min)
check_runtime           % runs every Section 4.3 method once on a small network (~30 s)
```

`check_accuracy` should print `MATCH`. The stored results come from R2023a. In settings where no method finds the groups (errors near 60%, for example n = 100 and ρ = 0.05), k-means can end in a different partition on another MATLAB version, so single networks there may differ. Running the original code and this code on the same MATLAB version gives identical results.

Don't run the Section 4.3 settings on a laptop. They need 2–8 GiB per MATLAB process, and the UASE-based method takes up to 3 minutes per fit.

## 3. Rerun the simulations on a Slurm cluster

First edit `--account`, `--constraint` and the `module load matlab/...` line in the `.sbatch` files for your cluster. New results go to `simulations/output/` (git-ignored), so they never overwrite `results/`.

### Section 4.2: accuracy (Figures 1–2, Table 1)

`accuracy/settings.csv` has 208 settings: n = 100, 125, …, 400; L = 100, 150, 200, 250; and ρ = 0.05, 0.10, 0.15, 0.20. Each setting gets 100 networks. For each network, `run_setting.m` records:

- Algorithm 1 with the threshold n_l ≥ log n/ρ̂;
- Algorithm 1 at every fixed threshold β⋆ = 0.30, 0.35, …, 1.10;
- all-layer clustering;
- the UASE-based method.

Each setting took 1–21 minutes on one core (median 3.4).

```bash
cd simulations/accuracy
sbatch --array=1-208%40 run.sbatch
./status.sh                   # wait for 208 / 208
sbatch summarize.sbatch       # writes ../output/accuracy/summary/{curves,settings_summary,networks}.csv
```

If a task stops, resubmit it with the same id. `run_setting` saves after every network and continues where it stopped.

### Section 4.3: running time and memory (Table 2, Figures 3–4)

`runtime/settings.csv` has the 9 settings below. The ids fix the seeds, which is why they are not numbered 1–9.

| id | n | L | ρ | In the paper |
|---|---|---|---|---|
| 6 | 3000 | 1000 | 0.05 | Table 2 (first block), Figure 4 |
| 12 | 3000 | 1000 | 0.01 | Table 2 (second block), Figure 3 setting 4 |
| 13 | 3000 | 3000 | 0.01 | Table 2 (third block), Figure 3 setting 1 |
| 14 | 2000 | 3000 | 0.01 | Figure 3 setting 2 |
| 15 | 4000 | 3000 | 0.01 | Figure 3 setting 3 |
| 16 | 3000 | 2000 | 0.01 | Figure 3 setting 5 |
| 17 | 3000 | 4000 | 0.01 | Figure 3 setting 6 |
| 18 | 3000 | 3000 | 0.005 | Figure 3 setting 7 |
| 19 | 3000 | 3000 | 0.02 | Figure 3 setting 8 |

How it is measured (`run_network.sh`, `bench_one.m`):

- Each network is generated and saved before it is timed. Generation is not timed. Network files take 120–520 MB and are deleted after use.
- Each method runs in its own MATLAB process with one thread.
- The process loads the network, warms up on a small network, and then runs three full fits. The time is the mean of the three. For Algorithm 1 it includes computing the threshold.
- The memory is the peak memory of the whole MATLAB process, from `/usr/bin/time`. The "no method" process (`Input_only`) only loads the network.
- A setting reports the average error and the median time and memory over its 10 networks.
- Each array task has a whole Cascade Lake node to itself.

```bash
cd simulations/runtime
sbatch --array=6,12-19 bench.sbatch    # Table 2 and Figure 3: 4 methods x 10 networks per setting (1-4 h per setting)
sbatch --array=1-10 sweep.sbatch       # Figure 4: 12 methods on each network of setting 6 (~20 min per network)
./status.sh
python3 summarize.py ../output/runtime         # writes runs.csv and summary.csv
python3 summarize.py ../output/active_sweep
```

Resubmitting a task skips the runs that already finished.

### Compare with the paper

```bash
cd simulations/figures
mkdir -p ../output/results && cp -r ../output/accuracy/summary ../output/results/accuracy
cp -r ../output/runtime ../output/results/runtime && cp -r ../output/active_sweep ../output/results/active_sweep
python3 make_tables.py ../output/results
python3 make_figures.py ../output/results ../output/figures
```

Running times depend on the processor and on the load of the node, so they will not match to the second. Errors use fixed seeds and should match.

## Model and methods

All settings use M = 3 groups of layers and K = 3 communities.

**Network** (`lib/generate_network.m`):

- Layer widths are n_l = max(K, ⌊n^β_l⌋) with β_l ~ U[0.30, 1.10].
- Each layer is assigned to a group at random.
- In each group, the n rows are split at random into K communities whose sizes differ by at most one. The same holds for the n_l columns of each layer.
- B^(l) = 2ρ·U with U ~ U(0,1)^{K×K}. U is drawn again while σ₁(U)/σ_K(U) > 10.
- A^(l)(i,j) ~ Bernoulli(P^(l)(i,j)).

**Seeds:**

- Network r of accuracy setting id uses seed 6000000 + 1000·id + r.
- Network r of runtime setting id uses seed 8000000 + 1000·id + r.
- Every eigenvector computation and every k-means call is seeded from the network seed, so the results don't depend on the order of the computations.

**Algorithm 1** (`lib/fit_method.m`, and `algorithm1` in `accuracy/run_setting.m`):

- A layer is active if n_l ≥ log n/ρ̂, where ρ̂ is the fraction of ones over all adjacency matrices.
- **Stage 1** runs on the active layers:
  - U_l are the K leading eigenvectors of A_l A_lᵀ − diag(A_l A_lᵀ);
  - T(i,j) = ‖U_iᵀU_j‖²_F;
  - k-means (10 starts) on the M leading eigenvectors of T.
- **Stage 2** assigns each non-active layer to the group m that maximizes ‖(Π_⊥ Û^(m))ᵀ A_l‖²_F. Û^(m) are the K leading eigenvectors of the summed hollowed Gram matrix of the active layers in group m, and Π_⊥ removes the column means.
- The two implementations give the same labels:
  - `accuracy/run_setting.m` computes T once per network and reuses it for all 17 fixed thresholds;
  - `lib/fit_method.m`, which is timed, uses matrix–vector products, so the n×n matrices are never formed.

**All-layer clustering** is Stage 1 on all layers.

**UASE-based method:**

- V are the KM leading left singular vectors of the unfolded matrix [A_1, …, A_L].
- Each layer gets the feature vec(Vᵀ(A_l A_lᵀ − diag(A_l A_lᵀ))V), scaled to unit norm.
- The L features are clustered by spectral clustering: M leading left singular vectors, rows normalized, then k-means.

**Errors** are fractions of layers, minimized over the M! relabelings of the groups:

- the total error is over all layers;
- the clustering error is over the active layers;
- the classification error is over the non-active layers.

## Where the stored results come from

The stored results were computed in several cluster runs between Nov 5 and Nov 10, 2026. This repository merges the code of those runs. The merged code gives the same labels as each original run, which was checked on the same MATLAB version.

- **`results/accuracy`:**
  - The fixed-threshold curves and the UASE-based errors come from the Nov 5 run.
  - Algorithm 1 with the threshold n_l ≥ log n/ρ̂ comes from the Nov 7 run, on the same networks. All-layer clustering was computed in both runs, with identical results.
  - The `seconds` column is from the Nov 5 run.
- **`results/runtime`:**
  - Settings 13–19 are from the Nov 7 run.
  - For settings 6 and 12, Algorithm 1, all-layer clustering and the no-method process are from the Nov 7 run. The UASE-based method is from an earlier run on the same networks and the same type of node.
- **`results/active_sweep`:** the Nov 10 run.
