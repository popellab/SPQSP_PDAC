#!/bin/bash
#SBATCH --account=bgre-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=sbi_bench
#SBATCH --output=bench-%j.out

set -eo pipefail

module load pytorch-conda/2.8
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate base

cd ~/sbi_bench

nvidia-smi -L || true
echo "---"

# GPU run
"$HOME/sbi-env/bin/python" bench_npe.py \
  --theta theta_train.pt --x x_train.pt --config npe_config.json \
  --device cuda --batch-sizes 128,512,2048,8192 --iters 20 --sbi-epochs 1 \
  --output bench_gpu.json

echo "---"

# CPU run on same machine (for a clean GPU-vs-CPU comparison on same hardware)
"$HOME/sbi-env/bin/python" bench_npe.py \
  --theta theta_train.pt --x x_train.pt --config npe_config.json \
  --device cpu --batch-sizes 128,512,2048 --iters 10 --sbi-epochs 1 \
  --output bench_cpu.json
