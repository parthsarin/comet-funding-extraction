#!/bin/bash
#SBATCH --job-name=sft-qwen3.5
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sft_qwen3.5_%j.out
#SBATCH --error=logs/sft_qwen3.5_%j.err

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: sft-qwen3.5-9B"
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

SFT_DATA="data/sft_glm_distill.jsonl"
if [ ! -s "${SFT_DATA}" ]; then
    echo "ERROR: SFT data file missing or empty: ${SFT_DATA}"
    exit 1
fi
echo "SFT data: $(wc -l < ${SFT_DATA}) samples"
echo ""

echo "Starting Qwen3.5-9B SFT (distillation)..."
python train_sft.py \
    --model Qwen/Qwen3.5-9B \
    --sft-data-path "${SFT_DATA}" \
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
    echo "Qwen3.5-9B SFT completed!"
    echo "End Time: $(date)"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "ERROR: Qwen3.5-9B SFT failed!"
    echo "End Time: $(date)"
    echo "========================================="
    exit 1
fi
