#!/bin/bash
#SBATCH --job-name=rescore-glm
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rescore_glm_%j.out
#SBATCH --error=logs/rescore_glm_%j.err

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: rescore-glm"
echo "Node: ${SLURMD_NODENAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Start Time: $(date)"
echo "========================================="
echo ""

cd /scratch/m000152-pm05/arxiv-funding-statements
mkdir -p logs results data

source /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh
conda activate comet-inference

echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# =========================================
# Step 1: Re-score with fixed max_tokens
# =========================================
RAW_PATH="results/glm_samples_181475_raw.json"
SCORED_PATH="results/glm_scored_${SLURM_JOB_ID}.json"

echo "Step 1: Re-scoring GLM samples with max_tokens=2048..."
echo "  Input: ${RAW_PATH}"
echo "  Output: ${SCORED_PATH}"

python score_with_llm.py \
    --input-path "${RAW_PATH}" \
    --output-path "${SCORED_PATH}" \
    --model zai-org/GLM-4.5-Air \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9

if [ $? -ne 0 ]; then
    echo "ERROR: Re-scoring failed!"
    exit 1
fi
echo "Re-scoring complete: ${SCORED_PATH}"
echo ""

# =========================================
# Step 2: Build SFT dataset
# =========================================
echo "Step 2: Building SFT dataset..."
python build_sft_dataset.py \
    --scored-path "${SCORED_PATH}" \
    --data-path data/train.jsonl \
    --output-path data/sft_glm_distill.jsonl \
    --score-threshold 0.7

if [ $? -ne 0 ]; then
    echo "ERROR: SFT dataset build failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "Re-scoring pipeline completed!"
echo "  Scored: ${SCORED_PATH}"
echo "  SFT data: data/sft_glm_distill.jsonl"
echo "End Time: $(date)"
echo "========================================="
