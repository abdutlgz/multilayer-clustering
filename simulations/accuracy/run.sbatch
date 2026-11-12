#!/usr/bin/env bash
# Section 4.2: one array task = one setting (row id of settings.csv), 100 networks.
# Submit from simulations/accuracy:   sbatch --array=1-208%40 run.sbatch
# Edit --account and the module line for your cluster.
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=accuracy
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load matlab/matlab-R2023a
OUT="${OUT:-$(cd .. && pwd)/output/accuracy}"; mkdir -p "$OUT"
export MATLAB_PREFDIR="${SLURM_TMPDIR:-/tmp}/accuracy_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MATLAB_PREFDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
matlab -singleCompThread -batch "run_setting(${SLURM_ARRAY_TASK_ID},'$OUT',${REPS:-100})"
rm -rf "$MATLAB_PREFDIR"
