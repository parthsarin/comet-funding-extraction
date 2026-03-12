#!/bin/bash
#SBATCH --job-name=debug-load
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=0:30:00
#SBATCH --output=logs/debug_load_%j.out
#SBATCH --error=logs/debug_load_%j.err

cd /scratch/m000152-pm05/arxiv-funding-statements
source /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh
conda activate comet-inference

echo "Start: $(date)"
echo "Memory limit: $(ulimit -v)"
echo ""

python debug_load.py

echo ""
echo "Exit code: $?"
echo "Done: $(date)"
