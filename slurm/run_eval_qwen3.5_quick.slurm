#!/bin/bash
#SBATCH --job-name=eval-qwen3.5
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_qwen3.5_quick_%j.out
#SBATCH --error=logs/eval_qwen3.5_quick_%j.err

cd /scratch/m000152-pm05/arxiv-funding-statements
source /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh
conda activate comet-inference

echo "Start: $(date)"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

python eval_with_transformers.py \
    --model checkpoints/qwen3.5-9b-sft-distill-merged \
    --data-path data/test.jsonl \
    --output-path results/eval_qwen3.5_distill_nothink_${SLURM_JOB_ID}.json \
    --max-tokens 2048 \
    --verbose

echo "Done: $(date)"
