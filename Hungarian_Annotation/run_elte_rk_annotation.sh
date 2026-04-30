#!/bin/bash
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --qos=normal
#SBATCH --job-name=elte_rk
#SBATCH --output=logs/annotate_%x_%j.out
#SBATCH --error=logs/annotate_%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/venv-huspacy/bin/activate}"

LOG_DIR="${PROJECT_ROOT}/logs"
ANNOTATE_SCRIPT="${PROJECT_ROOT}/Hungarian_Annotation/annotate_data.py"

INPUT_DIR="${PROJECT_ROOT}/Elte_Rk_Sentences"
OUTPUT_DIR="${PROJECT_ROOT}/Elte_Rk_Annotated_Sentences"
DICT_PATH="${PROJECT_ROOT}/New_Data_Extractor/eksz_definitions_only.json"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Project root: $PROJECT_ROOT"
echo "=========================================="

cd "$PROJECT_ROOT"
source "$VENV_PATH"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 "$ANNOTATE_SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dict_path "$DICT_PATH" \
    --max_workers 30 \
    --chunk_size 500 \
    --model_name hu_core_news_lg

echo "=========================================="
echo "Finished at $(date)"
echo "=========================================="