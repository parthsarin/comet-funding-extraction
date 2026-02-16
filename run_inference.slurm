#!/bin/bash
#SBATCH --job-name=funding-extract
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# Print job information
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: funding-extract"
echo "Node: ${SLURMD_NODENAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Start Time: $(date)"
echo "========================================="
echo ""

# Change to working directory
cd /scratch/m000152-pm05/arxiv-funding-statements

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate comet-inference

# Print environment info
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo ""

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run inference
echo "Starting funding extraction inference..."
python sample.py \
    --data-path data/train.jsonl \
    --output-path results/inference_results_${SLURM_JOB_ID}.json \
    --model Qwen/Qwen3-8B \
    --k 5 \
    --temperature 1.0 \
    --max-tokens 2048 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --verbose

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Inference completed successfully!"
    echo "Results saved to: results/inference_results_${SLURM_JOB_ID}.json"
    echo "End Time: $(date)"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "ERROR: Inference failed!"
    echo "End Time: $(date)"
    echo "========================================="
    exit 1
fi
