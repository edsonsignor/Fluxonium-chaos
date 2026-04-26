#!/bin/bash
#SBATCH --job-name=lyapunov-map
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/lyapunov_maps

# Edit this section to match your cluster's Python setup.
# Examples:
#   module load python/3.11
#   source .venv/bin/activate
#   source ~/venvs/fluxonium/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

WORKERS="${SLURM_CPUS_PER_TASK:-1}"
OUTFILE="data/lyapunov_maps/lyapunov_map_${SLURM_JOB_ID}.h5"
EC="1.0"
EJ="3.6"
EL="0.7"
PHI_EXT0="3.141592653589793"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Workers: ${WORKERS}"
echo "Output: ${OUTFILE}"
echo "EC=${EC}, EJ=${EJ}, EL=${EL}, phi_ext0=${PHI_EXT0}"
echo "Start time: $(date)"

python scripts/lyapunov_map.py \
    --workers "${WORKERS}" \
    --outfile "${OUTFILE}" \
    --EC "${EC}" \
    --EJ "${EJ}" \
    --EL "${EL}" \
    --phi-ext0 "${PHI_EXT0}"

echo "End time: $(date)"
