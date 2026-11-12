# Simulations (Section 4)

## What this folder shows

We generate random multilayer networks in which the true groups of layers are known. Then we run three methods on them and count how many layers each method puts in the wrong group:

- **Algorithm 1**, our method: it groups the *active* layers (those with enough data) and then assigns the other layers to the groups;
- **all-layer spectral clustering**, which groups all layers at once;
- the **UASE-based method**, which also groups all layers at once.

The **error** of a method is the fraction of layers in the wrong group, so lower is better.

There are two experiments:

| Experiment | Paper | Networks | Question |
|---|---|---|---|
| Accuracy | Section 4.2 | 208 settings of small networks, 100 networks each | How accurate is each method, and how should "active" be defined? |
| Speed and memory | Section 4.3 | 9 settings of large networks, 10 networks each | How long does each method take, and how much memory does it need? |

## Where each figure and table comes from

| In the paper | What it shows | Data file in `results/` | Made by |
|---|---|---|---|
| Figure 1 | error when the threshold for "active" is fixed by hand | `accuracy/curves.csv`, `accuracy/settings_summary.csv` | `figures/make_figures.py` |
| Table 1 | number of active layers and the errors of Algorithm 1 | `accuracy/settings_summary.csv` | `figures/make_tables.py` |
| Figure 2 | error of the three methods as the network grows | `accuracy/settings_summary.csv` | `figures/make_figures.py` |
| Table 2 | error, time and memory in three large settings | `runtime/summary.csv` | `figures/make_tables.py` |
| Figure 3 | error, time and memory in eight large settings | `runtime/summary.csv` | `figures/make_figures.py` |
| Figure 4 | what happens when more or fewer layers are made active | `active_sweep/runs.csv` | `figures/make_figures.py` |

The figure files are in `results/figures/`. They are identical to the figures in the paper.

## Step 1: rebuild the figures and tables (a few seconds)

You need Python 3 with numpy, pandas and matplotlib. You do not need MATLAB for this step.

```bash
cd simulations/figures
python3 make_figures.py        # writes Figures 1-4 to ../results/figures/
python3 make_tables.py         # prints Tables 1-2 and the numbers quoted about Figures 3-4
```

## Step 2: check the code on a laptop (under 2 minutes)

You need MATLAB with the Statistics and Machine Learning Toolbox. In MATLAB, go to the `simulations/check` folder and run:

```matlab
check_accuracy          % reruns 10 networks of one setting and compares them with results/  (~15 s)
check_accuracy(72,100)  % all 100 networks of that setting, including its Figure 1 curve           (~1.5 min)
check_runtime           % runs every method once on a small network                                 (~30 s)
```

`check_accuracy` should print `MATCH`.

Two cautions:

- The stored results come from MATLAB R2023a. In settings where no method can find the groups (errors near 60%, for example n = 100 and ρ = 0.05), another MATLAB version can end k-means in a different grouping, so a single network may differ there.
- Do not run the large settings of Step 3 on a laptop. They need 2–8 GB of memory per MATLAB process, and the UASE-based method takes up to 3 minutes per fit.

## Step 3: rerun everything on a computing cluster (many hours)

The scripts are written for a Slurm cluster. First edit the `--account`, `--constraint` and `module load matlab/...` lines in the `.sbatch` files for your cluster. New results go to `simulations/output/`, so they never overwrite `results/`. If a job stops, submit it again: finished work is skipped.

### 3a. Accuracy (Figures 1–2, Table 1)

```bash
cd simulations/accuracy
sbatch --array=1-208%40 run.sbatch    # one job per setting; each takes 1-21 minutes on one core
./status.sh                           # wait until it shows 208 / 208
sbatch summarize.sbatch               # writes ../output/accuracy/summary/
```

### 3b. Speed and memory (Table 2, Figures 3–4)

```bash
cd simulations/runtime
sbatch --array=6,12-19 bench.sbatch   # Table 2 and Figure 3: 1-4 hours per setting
sbatch --array=1-10 sweep.sbatch      # Figure 4: about 20 minutes per network
./status.sh
python3 summarize.py ../output/runtime
python3 summarize.py ../output/active_sweep
```

### 3c. Compare with the paper

```bash
cd simulations/figures
mkdir -p ../output/results && cp -r ../output/accuracy/summary ../output/results/accuracy
cp -r ../output/runtime ../output/results/runtime && cp -r ../output/active_sweep ../output/results/active_sweep
python3 make_tables.py ../output/results
python3 make_figures.py ../output/results ../output/figures
```

The errors use fixed random seeds and should match. Running times depend on the processor and on how busy it is, so they will not match to the second.

## What is in each folder

```
lib/        code shared by both experiments
  generate_network.m      makes one random network (Section 4.1)
  select_active_layers.m  the rule for "active": n_l >= log n / rho_hat
  fit_method.m            Algorithm 1, all-layer, UASE-based, and Algorithm 1 with the widest XX% of layers active
  leading_space.m         leading singular vectors (with eigs or irlba)
  layer_errors.m          total, clustering and classification error
  irlba.m                 third-party code by J. Baglama and L. Reichel (see the file header)
accuracy/   Section 4.2 (Figures 1-2, Table 1); settings.csv lists the 208 settings
runtime/    Section 4.3 (Table 2, Figures 3-4); settings.csv lists the 9 settings
figures/    make_figures.py and make_tables.py
check/      the short checks of Step 2
results/    the results used in the paper (CSV files) and the figures
```

## Technical details

This part is for readers who want the exact settings.

### Requirements

- MATLAB with the Statistics and Machine Learning Toolbox. The paper results were computed with R2023a on the UCF Stokes cluster (Slurm); the timing runs used Intel Xeon Gold 6240R (Cascade Lake) nodes.
- Python 3 with pandas, numpy and matplotlib for the figures (tested with Python 3.9, pandas 2.3, matplotlib 3.9). `runtime/summarize.py` needs only the standard library.
- Linux with GNU `/usr/bin/time` and `lscpu`, for the memory measurements in `runtime/`.

### The simulated networks (`lib/generate_network.m`)

All settings use M = 3 groups of layers and K = 3 communities.

- Layer l has n rows and n_l = max(K, ⌊n^β_l⌋) columns, with β_l drawn uniformly from [0.30, 1.10].
- Each layer is assigned to one of the 3 groups at random.
- In each group, the n rows are split at random into K communities whose sizes differ by at most one. The same holds for the n_l columns of each layer.
- B^(l) = 2ρ·U with U ~ U(0,1)^{K×K}. U is drawn again while σ₁(U)/σ_K(U) > 10.
- A^(l)(i,j) ~ Bernoulli(P^(l)(i,j)).

The accuracy settings (`accuracy/settings.csv`) are all combinations of n = 100, 125, …, 400; L = 100, 150, 200, 250; and ρ = 0.05, 0.10, 0.15, 0.20. For each network, `accuracy/run_setting.m` records Algorithm 1 with the rule n_l ≥ log n/ρ̂, Algorithm 1 at every fixed threshold β⋆ = 0.30, 0.35, …, 1.10, all-layer clustering, and the UASE-based method.

The speed and memory settings (`runtime/settings.csv`). The ids fix the seeds, which is why they are not numbered 1–9:

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

### Random seeds

- Network r of accuracy setting `id` uses seed 6000000 + 1000·id + r.
- Network r of speed and memory setting `id` uses seed 8000000 + 1000·id + r.
- Every eigenvector computation and every k-means call is seeded from the network seed, so the results do not depend on the order of the computations.

### The methods

**Algorithm 1** (`lib/fit_method.m`, and `algorithm1` in `accuracy/run_setting.m`):

- A layer is active if n_l ≥ log n/ρ̂, where ρ̂ is the fraction of ones over all adjacency matrices.
- **Stage 1** runs on the active layers:
  - U_l are the K leading eigenvectors of A_l A_lᵀ − diag(A_l A_lᵀ);
  - T(i,j) = ‖U_iᵀU_j‖²_F;
  - k-means (10 starts) on the M leading eigenvectors of T.
- **Stage 2** assigns each non-active layer to the group m that maximizes ‖(Π_⊥ Û^(m))ᵀ A_l‖²_F. Û^(m) are the K leading eigenvectors of the summed hollowed Gram matrix of the active layers in group m, and Π_⊥ removes the column means.
- The two implementations give the same labels. `accuracy/run_setting.m` computes T once per network and reuses it for all 17 fixed thresholds. `lib/fit_method.m`, which is the one timed, uses matrix–vector products, so the n×n matrices are never formed.

**All-layer clustering** is Stage 1 on all layers.

**UASE-based method:**

- V are the KM leading left singular vectors of the unfolded matrix [A_1, …, A_L].
- Each layer gets the feature vec(Vᵀ(A_l A_lᵀ − diag(A_l A_lᵀ))V), scaled to unit norm.
- The L features are clustered by spectral clustering: M leading left singular vectors, rows normalized, then k-means.

**Errors** are fractions of layers, minimized over the M! relabelings of the groups:

- the **total error** is over all layers;
- the **clustering error** is over the active layers;
- the **classification error** is over the non-active layers.

### How time and memory are measured (`runtime/run_network.sh`, `runtime/bench_one.m`)

- Each network is generated and saved before it is timed. Generation is not timed. Network files take 120–520 MB and are deleted after use.
- Each method runs in its own MATLAB process with one thread.
- The process loads the network, warms up on a small network, and then runs three full fits. The time is the mean of the three. For Algorithm 1 it includes computing the threshold.
- The memory is the peak memory of the whole MATLAB process, from `/usr/bin/time`. The "no method" process (`Input_only`) only loads the network.
- A setting reports the average error and the median time and memory over its 10 networks.
- Each array job has a whole Cascade Lake node to itself.

### Where the stored results come from

The stored results were computed in several cluster runs between Nov 5 and Nov 10, 2026. This folder merges the code of those runs. The merged code gives the same labels as each original run, which was checked on the same MATLAB version.

- **`results/accuracy`:**
  - The fixed-threshold curves and the UASE-based errors come from the Nov 5 run.
  - Algorithm 1 with the rule n_l ≥ log n/ρ̂ comes from the Nov 7 run, on the same networks. All-layer clustering was computed in both runs, with identical results.
  - The `seconds` column is from the Nov 5 run.
- **`results/runtime`:**
  - Settings 13–19 are from the Nov 7 run.
  - For settings 6 and 12, Algorithm 1, all-layer clustering and the no-method process are from the Nov 7 run. The UASE-based method is from an earlier run on the same networks and the same type of node.
- **`results/active_sweep`:** the Nov 10 run.
