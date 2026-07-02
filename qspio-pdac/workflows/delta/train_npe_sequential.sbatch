#!/bin/bash
#SBATCH --account=bgre-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=sbi_train_seq
#SBATCH --output=sbi_train_seq-%j.out

set -eo pipefail

module load pytorch-conda/2.8
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate base

RUN_DIR="${RUN_DIR:-$PWD}"
cd "$RUN_DIR"

nvidia-smi | head -20 || true

"$HOME/sbi-env/bin/python" train_npe_sequential.py \
  --manifest manifest.json \
  --input-dir . \
  --output-dir out \
  --device cuda
