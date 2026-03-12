#!/bin/bash
#SBATCH --job-name=eval-base
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_base_%j.out
#SBATCH --error=logs/eval_base_%j.err

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: eval-base-llama"
echo "Start Time: $(date)"
echo "========================================="

cd /scratch/m000152-pm05/arxiv-funding-statements
mkdir -p logs results

source /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh
conda activate comet-inference

echo "Evaluating base LLaMA-3.1-8B-Instruct on test set..."
python sample.py \
    --data-path data/test.jsonl \
    --output-path results/eval_llama_base_${SLURM_JOB_ID}.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --k 1 \
    --temperature 0.0 \
    --max-tokens 4096 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --funding-statement-only \
    --verbose

echo "Done: $(date)"
