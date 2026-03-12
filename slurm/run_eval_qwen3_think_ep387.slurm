#!/bin/bash
#SBATCH --job-name=ev-qwen3-387
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_qwen3_think_ep387_%j.out
#SBATCH --error=logs/eval_qwen3_think_ep387_%j.err

cd /scratch/m000152-pm05/arxiv-funding-statements
source /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh
conda activate comet-inference

echo "Start: $(date)"
python eval_with_transformers.py \
    --model Qwen/Qwen3-8B \
    --adapter checkpoints/sft-Qwen3-8B-distill-20260304-0950/checkpoint-387 \
    --data-path data/test.jsonl \
    --output-path results/eval_qwen3_think_ep387.json \
    --enable-thinking \
    --max-tokens 4096 \
    --verbose
echo "Done: $(date)"
