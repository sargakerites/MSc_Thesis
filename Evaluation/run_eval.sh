#!/bin/bash
#SBATCH -J eval_ft_testsplit
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/venv/bin/activate}"

LOG_DIR="${PROJECT_ROOT}/Evaluation/logs"
OUTPUT_DIR="${PROJECT_ROOT}/Evaluation/Results/finetuned_testsplit"
SCRIPT_PATH="${PROJECT_ROOT}/Evaluation/evaluate.py"
MODEL_PATH="${SCRATCH_ROOT}/Fine-Tuning/Final/Checkpoints/final_bs1024_e6"
DATASET_PATH="${PROJECT_ROOT}/Training_Dataset_Split/test"

echo "Starting job on $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

source "$VENV_PATH"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export SACREBLEU="${SCRATCH_ROOT}/.sacrebleu"

export HF_HOME="${SCRATCH_ROOT}/.hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/.hf_datasets_cache"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Running evaluation..."

python "$SCRIPT_PATH" \
  --model "$MODEL_PATH" \
  --dataset "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 8 \
  --max_source_length 256 \
  --max_target_length 256 \
  --num_beams 4 \
  --local_files_only

echo "Evaluation finished."