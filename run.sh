#!/bin/bash
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100_80gb:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/%u/multimodal_multilingual_translator/logs/%j.log
#SBATCH --job-name=multimodel-multilingual-translate
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL

export HF_TOKEN=$(grep -v '^#' .env | grep 'HUGGINGFACE_TOKEN=' | cut -d '=' -f2-)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TMPDIR="/tmp"
export TMP="/tmp"
export TEMP="/tmp"

export SCRATCH="${SCRATCH:-/scratch/alpine/$USER}"

export HF_HOME="$SCRATCH/.cache/huggingface"
export EVALUATE_CACHE_DIR="$SCRATCH/.cache/evaluate"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/transformers"

mkdir -p "$HF_HOME" "$EVALUATE_CACHE_DIR" "$TRANSFORMERS_CACHE"

module purge

# 1. Properly initialize conda for non-interactive Slurm shell

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /projects/$USER/multimodal_multilingual_translator

module load uv
uv sync
uv run run.py "$1"
