#!/bin/bash
#SBATCH --job-name=eval-think-all
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_thinking_all_%j.out
#SBATCH --error=logs/eval_thinking_all_%j.err

cd /scratch/m000152-pm05/arxiv-funding-statements
source /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh
conda activate comet-inference

echo "Start: $(date)"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Qwen3.5-9B thinking checkpoints (epochs 1-3)
for CKPT in 129 258 387; do
    echo "========================================="
    echo "Evaluating Qwen3.5-9B thinking checkpoint-${CKPT}"
    echo "Time: $(date)"
    echo "========================================="
    python eval_with_transformers.py \
        --model Qwen/Qwen3.5-9B \
        --adapter checkpoints/sft-Qwen3.5-9B-distill-20260304-0948/checkpoint-${CKPT} \
        --data-path data/test.jsonl \
        --output-path results/eval_qwen3.5_think_ep${CKPT}.json \
        --enable-thinking \
        --max-tokens 4096 \
        --verbose
    echo ""
done

# Qwen3-8B thinking checkpoints (epochs 1-3)
for CKPT in 129 258 387; do
    echo "========================================="
    echo "Evaluating Qwen3-8B thinking checkpoint-${CKPT}"
    echo "Time: $(date)"
    echo "========================================="
    python eval_with_transformers.py \
        --model Qwen/Qwen3-8B \
        --adapter checkpoints/sft-Qwen3-8B-distill-20260304-0950/checkpoint-${CKPT} \
        --data-path data/test.jsonl \
        --output-path results/eval_qwen3_think_ep${CKPT}.json \
        --enable-thinking \
        --max-tokens 4096 \
        --verbose
    echo ""
done

echo "All evaluations done: $(date)"
