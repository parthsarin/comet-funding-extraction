#!/bin/bash
#SBATCH --job-name=eval-qwen3.5
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm05
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_qwen3.5_distill_%j.out
#SBATCH --error=logs/eval_qwen3.5_distill_%j.err

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: eval-qwen3.5-distill"
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

# Find the latest Qwen3.5-9B distill checkpoint
ADAPTER_DIR=$(ls -td checkpoints/sft-Qwen3.5-9B-distill-*/final 2>/dev/null | head -1)
if [ -z "$ADAPTER_DIR" ]; then
    ADAPTER_DIR=$(ls -td checkpoints/sft-Qwen3.5-9B-distill-*/checkpoint-* 2>/dev/null | head -1)
fi
if [ -z "$ADAPTER_DIR" ]; then
    echo "ERROR: No Qwen3.5-9B distill checkpoint found in checkpoints/"
    ls -la checkpoints/
    exit 1
fi
echo "Using adapter: ${ADAPTER_DIR}"

MERGED_DIR="checkpoints/qwen3.5-9b-sft-distill-merged"

# Step 1: Merge LoRA
echo "Merging LoRA adapter..."
python merge_lora.py \
    --base-model Qwen/Qwen3.5-9B \
    --adapter-path "${ADAPTER_DIR}" \
    --output-path "${MERGED_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: LoRA merge failed!"
    exit 1
fi

# Patch config for vLLM compatibility (vLLM 0.11 lacks native Qwen3.5 support)
echo "Patching config.json for vLLM TransformersForCausalLM fallback..."
python -c "
import json
cfg_path = '${MERGED_DIR}/config.json'
with open(cfg_path) as f:
    cfg = json.load(f)
cfg['architectures'] = ['TransformersForCausalLM']
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print('Patched architectures to TransformersForCausalLM')
"

# Step 2: Evaluate on test set (using transformers, not vLLM)
echo ""
echo "Evaluating merged Qwen3.5-9B distill model on test set..."
python eval_with_transformers.py \
    --model "${MERGED_DIR}" \
    --data-path data/test.jsonl \
    --output-path results/eval_qwen3.5_distill_${SLURM_JOB_ID}.json \
    --max-tokens 4096 \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Qwen3.5-9B distill evaluation completed!"
    echo "Results: results/eval_qwen3.5_distill_${SLURM_JOB_ID}.json"
    echo "End Time: $(date)"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "ERROR: Qwen3.5-9B distill evaluation failed!"
    echo "End Time: $(date)"
    echo "========================================="
    exit 1
fi
