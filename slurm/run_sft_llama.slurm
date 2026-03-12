#!/bin/bash
#SBATCH --job-name=sft-llama
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sft_llama_%j.out
#SBATCH --error=logs/sft_llama_%j.err

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: sft-llama"
echo "Node: ${SLURMD_NODENAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Start Time: $(date)"
echo "========================================="
echo ""

cd /scratch/m000152-pm05/arxiv-funding-statements
mkdir -p logs checkpoints

source /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh
conda activate comet-inference

echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

echo "Starting LLaMA-3.1-8B-Instruct SFT (ground truth)..."
python train_sft.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --ground-truth-path data/train.jsonl \
    --learning-rate 2e-5 \
    --num-epochs 3 \
    --per-device-batch-size 2 \
    --gradient-accumulation-steps 4 \
    --max-seq-length 4096 \
    --lora-r 16 \
    --lora-alpha 32 \
    --output-dir checkpoints \
    --report-to wandb

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "LLaMA SFT completed successfully!"
    echo "End Time: $(date)"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "ERROR: LLaMA SFT failed!"
    echo "End Time: $(date)"
    echo "========================================="
    exit 1
fi
