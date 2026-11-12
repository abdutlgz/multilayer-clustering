#!/usr/bin/env bash
# Section 4.3, Table 2 and Figure 3: one array task = one setting (id in settings.csv),
# 10 networks. For each network: generate it, run each method in a fresh single-thread
# MATLAB process (/usr/bin/time records its peak memory), then delete the network file
# (120-520 MB each).
# Submit from simulations/runtime:   sbatch --array=6,12-19 bench.sbatch
# Options: REPS=10, METHODS="Input_only Algorithm1 All_layer UASE", OUT=../output/runtime
# Edit --account, --constraint and the module line for your cluster.
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --constraint=cascadelake
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=runtime
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load matlab/matlab-R2023a
ID=${SLURM_ARRAY_TASK_ID}; REPS=${REPS:-10}
METHODS=${METHODS:-"Input_only Algorithm1 All_layer UASE"}
OUT="${OUT:-$(cd .. && pwd)/output/runtime}"
./run_network.sh "$ID" "$(seq 1 "$REPS")" "$METHODS" "$OUT"
