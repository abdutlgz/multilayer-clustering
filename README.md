# Multilayer Network Clustering Experiments

Code and results for the experiments of the paper on clustering the layers of a multilayer network
and classifying the non-active layers (Algorithm 1). Each layer is a rectangular matrix
A^(l) of size n × n_l: all layers share the same n row nodes, and each has its own columns.

## Repository structure

```text
simulations/
  Simulation study (Section 4): MATLAB code, the results used in the paper,
  and the scripts that make its tables and figures.

realworldexample/
  Real-data example on MovieLens 25M (Section 5): Python code, the results
  used in the paper, and the scripts that make its table, figures and numbers.
```

Each folder has a README that says which file makes each table and figure, how to rebuild them
from the stored results in a few seconds, how to check the code quickly, and how to rerun everything.

## Quick start

```bash
cd simulations/figures
python3 make_figures.py    # Figures 1-4 of Section 4
python3 make_tables.py     # Tables 1-2 of Section 4

cd ../../realworldexample/figures
python3 make_tables.py     # Table 3 and the numbers of Section 5
python3 make_figures.py    # Figures 5-7 of Section 5
```

## Data policy

Raw MovieLens files are not included in this repository. `realworldexample/get_data.sh`
downloads MovieLens 25M from GroupLens (https://grouplens.org/datasets/movielens/25m/) and checks
its MD5; see `realworldexample/README.md`.
