#!/usr/bin/env bash
# Section 4.3, active-layer sweep (Figure 4): setting 6 (n=3000, L=1000, rho=0.05; the
# same 10 networks as the first block of Table 2). One array task = one network r.
# Methods: Algorithm 1 with the widest 10%, 20%, ..., 100% of the layers active,
# Algorithm 1 with the threshold, and the "no method" process.
# Submit from simulations/runtime:   sbatch --array=1-10 sweep.sbatch
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --constraint=cascadelake
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=sweep
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load matlab/matlab-R2023a
METHODS=${METHODS:-"Input_only Algorithm1 Top10 Top20 Top30 Top40 Top50 Top60 Top70 Top80 Top90 Top100"}
OUT="${OUT:-$(cd .. && pwd)/output/active_sweep}"
./run_network.sh 6 "${SLURM_ARRAY_TASK_ID}" "$METHODS" "$OUT"
