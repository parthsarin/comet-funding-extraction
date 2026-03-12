#!/bin/bash
#SBATCH --job-name=ev-q35-258
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_qwen3.5_think_ep258_%j.out
#SBATCH --error=logs/eval_qwen3.5_think_ep258_%j.err

cd /scratch/m000152-pm05/arxiv-funding-statements
source /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh
conda activate comet-inference

echo "Start: $(date)"
python eval_with_transformers.py \
    --model Qwen/Qwen3.5-9B \
    --adapter checkpoints/sft-Qwen3.5-9B-distill-20260304-0948/checkpoint-258 \
    --data-path data/test.jsonl \
    --output-path results/eval_qwen3.5_think_ep258.json \
    --enable-thinking \
    --max-tokens 4096 \
    --verbose
echo "Done: $(date)"
