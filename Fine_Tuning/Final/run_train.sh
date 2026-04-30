#!/bin/bash
#SBATCH -J nllb_train_final_bs1024_e6
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=96:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err


PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/venv/bin/activate}"

RUN_NAME="final_bs1024_e6"

LOG_DIR="${PROJECT_ROOT}/Fine_Tuning/Final/logs"
OUTPUT_DIR="${SCRATCH_ROOT}/Fine_Tuning/Final/Checkpoints/${RUN_NAME}"
TRAIN_SCRIPT="${PROJECT_ROOT}/Fine_Tuning/Final/train.py"


source "$VENV_PATH"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export HF_HOME="${SCRATCH_ROOT}/.hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/.hf_datasets_cache"
export VLLM_CACHE_ROOT="${SCRATCH_ROOT}/.vllm_cache"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_ROOT"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"


echo "Starting final fine-tuning run..."
echo "Project root: $PROJECT_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Node: $(hostname)"
echo "Time: $(date)"

nvidia-smi


accelerate launch --num_processes 4 \
  "$TRAIN_SCRIPT" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 6 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 128 \
  --learning_rate 3e-5 \
  --logging_steps 50 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --warmup_ratio 0.03 \
  --bf16 \
  --gradient_checkpointing \
  --max_source_length 256 \
  --max_target_length 256 \
  --generation_num_beams 5 \
  --seed 42


echo "Finished at: $(date)"