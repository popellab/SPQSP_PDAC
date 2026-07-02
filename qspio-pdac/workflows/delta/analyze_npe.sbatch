#!/bin/bash
#SBATCH --account=bgre-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=sbi_analyze
#SBATCH --output=analyze-%j.out

set -eo pipefail

module load pytorch-conda/2.8
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate base

cd "${RUN_DIR:-$PWD}"

nvidia-smi -L || true
echo "---"

"$HOME/sbi-env/bin/python" analyze_npe_delta.py \
  --input-dir input --output-dir output --device cuda
