#!/bin/bash
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --qos=normal
#SBATCH --job-name=nllb_eval_compare
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/venv/bin/activate}"

LOG_DIR="${PROJECT_ROOT}/Evaluation/logs"
SCRIPT_PATH="${PROJECT_ROOT}/Evaluation/evaluate_base_vs_finetuned_sentence_bleu.py"

echo "======================================"
echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "======================================"

mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT"
source "$VENV_PATH"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export HF_HOME="${SCRATCH_ROOT}/.hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/.hf_datasets_cache"

echo "CUDA devices:"
nvidia-smi

python3 "$SCRIPT_PATH"

echo "======================================"
echo "Job finished at $(date)"
echo "======================================"