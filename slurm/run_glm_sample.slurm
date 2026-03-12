#!/bin/bash
#SBATCH --job-name=glm-pipeline
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=8:00:00
#SBATCH --output=logs/glm_pipeline_%j.out
#SBATCH --error=logs/glm_pipeline_%j.err

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: glm-pipeline"
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

# =========================================
# Step 1: Sample from GLM-4.5-Air
# =========================================
echo "Step 1: Starting GLM-4.5-Air sampling..."
python sample.py \
    --data-path data/train.jsonl \
    --output-path results/glm_samples_${SLURM_JOB_ID}.json \
    --model zai-org/GLM-4.5-Air \
    --k 5 \
    --temperature 0.7 \
    --max-tokens 4096 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --funding-statement-only \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: GLM sampling failed!"
    exit 1
fi
echo "Sampling complete: results/glm_samples_${SLURM_JOB_ID}_raw.json"
echo ""

# =========================================
# Step 2: Score with GLM-4.5-Air as verifier
# =========================================
echo "Step 2: Scoring responses with GLM-4.5-Air..."
python score_with_llm.py \
    --input-path results/glm_samples_${SLURM_JOB_ID}_raw.json \
    --output-path results/glm_scored_${SLURM_JOB_ID}.json \
    --model zai-org/GLM-4.5-Air \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9

if [ $? -ne 0 ]; then
    echo "ERROR: GLM scoring failed!"
    exit 1
fi
echo "Scoring complete: results/glm_scored_${SLURM_JOB_ID}.json"
echo ""

# =========================================
# Step 3: Build SFT dataset
# =========================================
echo "Step 3: Building SFT dataset..."
python build_sft_dataset.py \
    --scored-path results/glm_scored_${SLURM_JOB_ID}.json \
    --data-path data/train.jsonl \
    --output-path data/sft_glm_distill.jsonl \
    --score-threshold 0.7

if [ $? -ne 0 ]; then
    echo "ERROR: SFT dataset build failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "GLM pipeline completed successfully!"
echo "  Samples: results/glm_samples_${SLURM_JOB_ID}.json"
echo "  Scored: results/glm_scored_${SLURM_JOB_ID}.json"
echo "  SFT data: data/sft_glm_distill.jsonl"
echo "End Time: $(date)"
echo "========================================="
