#!/bin/bash
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --job-name=en_hu_eval
#SBATCH --output=logs/en_hu_eval_%j.out
#SBATCH --error=logs/en_hu_eval_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/venv/bin/activate}"

LOG_DIR="${PROJECT_ROOT}/Evaluation/logs"
OUTPUT_DIR="${PROJECT_ROOT}/Evaluation/semantic_eval_outputs"
SCRIPT_PATH="${PROJECT_ROOT}/Evaluation/semantic_evaluation.py"

echo "======================================"
echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "======================================"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"
source "$VENV_PATH"

export PYTHONUNBUFFERED=1

python3 -u "$SCRIPT_PATH"

echo "======================================"
echo "Job finished at $(date)"
echo "======================================"