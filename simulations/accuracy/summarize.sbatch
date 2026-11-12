#!/usr/bin/env bash
# Section 4.2: collect output/accuracy/setting_###.mat into output/accuracy/summary/*.csv.
# Submit from simulations/accuracy after all 208 settings finish:   sbatch summarize.sbatch
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=accuracy-sum
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load matlab/matlab-R2023a
OUT="${OUT:-$(cd .. && pwd)/output/accuracy}"
export MATLAB_PREFDIR="${SLURM_TMPDIR:-/tmp}/accuracy_sum_${SLURM_JOB_ID}"; mkdir -p "$MATLAB_PREFDIR"
matlab -singleCompThread -batch "summarize('$OUT')"
