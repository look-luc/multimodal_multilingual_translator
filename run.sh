#!/bin/bash
#SBATCH --partition=artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100_80gb:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/projects/%u/Multilingual-multimodal-translation/logs/%j.log
#SBATCH --job-name=multimodel-multilingual-translate
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL
