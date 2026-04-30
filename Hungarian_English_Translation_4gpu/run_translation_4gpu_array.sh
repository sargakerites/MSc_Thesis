#!/bin/bash
#SBATCH -p ai
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --qos=normal
#SBATCH --job-name=1_M_translation
#SBATCH --output=logs/translation4g_%A_%a.out
#SBATCH --error=logs/translation4g_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/venv/bin/activate}"

LOG_DIR="${PROJECT_ROOT}/logs"
SCRIPT_PATH="${PROJECT_ROOT}/Hungarian_English_Translation_4gpu/parallel_corpus_generation.py"

INPUT_DIR="${PROJECT_ROOT}/1_M_Annotated_Sentences"
OUTPUT_DIR="${PROJECT_ROOT}/1_M_Translated_Sentences"

echo "======================================"
echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "JOB: ${SLURM_JOB_ID}"
echo "ARRAY TASK: ${SLURM_ARRAY_TASK_ID}"
echo "Project root: $PROJECT_ROOT"
echo "======================================"

mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT"
source "$VENV_PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

export HF_HOME="${SCRATCH_ROOT}/.hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export VLLM_CACHE_ROOT="${SCRATCH_ROOT}/.vllm_cache"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$VLLM_CACHE_ROOT"

BUNDLE_SIZE=4
START_IDX=$((SLURM_ARRAY_TASK_ID * BUNDLE_SIZE))
END_IDX=$((START_IDX + BUNDLE_SIZE))

python3 "$SCRIPT_PATH" \
  --input-folder "$INPUT_DIR" \
  --output-folder "$OUTPUT_DIR" \
  --start-idx "$START_IDX" \
  --end-idx "$END_IDX" \
  --batch-size 32

echo "Finished at $(date)"