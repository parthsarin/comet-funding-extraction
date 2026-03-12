#!/bin/bash
#SBATCH --job-name=eval-llama
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_llama_%j.out
#SBATCH --error=logs/eval_llama_%j.err

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: eval-llama"
echo "Node: ${SLURMD_NODENAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Start Time: $(date)"
echo "========================================="
echo ""

cd /scratch/m000152-pm05/arxiv-funding-statements
mkdir -p logs results

source /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh
conda activate comet-inference

echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Use epoch-1 checkpoint (best loss) or override with ADAPTER_DIR env var
ADAPTER_DIR="${ADAPTER_DIR:-$(ls -td checkpoints/sft-Meta-Llama-3.1-8B-Instruct-*/checkpoint-165 2>/dev/null | head -1)}"
if [ -z "$ADAPTER_DIR" ]; then
    # Fallback to final checkpoint
    ADAPTER_DIR=$(ls -td checkpoints/sft-Meta-Llama-3.1-8B-Instruct-*/final 2>/dev/null | head -1)
fi
if [ -z "$ADAPTER_DIR" ]; then
    echo "ERROR: No LLaMA SFT checkpoint found in checkpoints/"
    exit 1
fi
echo "Using adapter: ${ADAPTER_DIR}"

MERGED_DIR="checkpoints/llama-3.1-8b-sft-merged"

# Step 1: Merge LoRA
echo "Merging LoRA adapter..."
python merge_lora.py \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --adapter-path "${ADAPTER_DIR}" \
    --output-path "${MERGED_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: LoRA merge failed!"
    exit 1
fi

# Step 2: Evaluate on test set
echo ""
echo "Evaluating merged model on test set..."
python sample.py \
    --data-path data/test.jsonl \
    --output-path results/eval_llama_sft_${SLURM_JOB_ID}.json \
    --model "${MERGED_DIR}" \
    --k 1 \
    --temperature 0.0 \
    --max-tokens 4096 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --funding-statement-only \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "LLaMA evaluation completed successfully!"
    echo "Results: results/eval_llama_sft_${SLURM_JOB_ID}.json"
    echo "End Time: $(date)"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "ERROR: LLaMA evaluation failed!"
    echo "End Time: $(date)"
    echo "========================================="
    exit 1
fi
