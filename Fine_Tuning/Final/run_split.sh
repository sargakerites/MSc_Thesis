#!/bin/bash
#SBATCH -J split_data
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/venv/bin/activate}"

SCRIPT_PATH="${PROJECT_ROOT}/Fine_Tuning/Second-Test/Load_and_Split_the_Data.py"

source "$VENV_PATH"

mkdir -p "${PROJECT_ROOT}/logs"

python "$SCRIPT_PATH"