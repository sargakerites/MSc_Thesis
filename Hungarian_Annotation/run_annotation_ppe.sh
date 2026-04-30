#!/bin/bash
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --qos=normal
#SBATCH --job-name=annotate_hu
#SBATCH --output=logs/1_M_annotate_%x_%j.out
#SBATCH --error=logs/1_M_annotate_%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/venv-huspacy/bin/activate}"

LOG_DIR="${PROJECT_ROOT}/logs"
EXTRACT_SCRIPT="${PROJECT_ROOT}/Hungarian_Annotation/extract_hungarian_sentences.py"
ANNOTATE_SCRIPT="${PROJECT_ROOT}/Hungarian_Annotation/annotate_data.py"

DATA_DIR="${PROJECT_ROOT}/data/news"
SENTENCE_DIR="${PROJECT_ROOT}/1_M_Sentences"
ANNOTATED_DIR="${PROJECT_ROOT}/1_M_Annotated_Sentences"
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

python3 "$EXTRACT_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$SENTENCE_DIR" \
    --chunk_size 500 \
    --max_files 1000 \
    --max_sentences 1000000

python3 "$ANNOTATE_SCRIPT" \
    --input_dir "$SENTENCE_DIR" \
    --output_dir "$ANNOTATED_DIR" \
    --dict_path "$DICT_PATH" \
    --max_workers 30 \
    --chunk_size 500 \
    --model_name hu_core_news_lg

echo "=========================================="
echo "Finished at $(date)"
echo "=========================================="