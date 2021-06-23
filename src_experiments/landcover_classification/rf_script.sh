#!/bin/bash

#SBATCH --partition=lotus_gpu

#SBATCH --account=lotus_gpu

#SBATCH --gres=gpu:1 # Request a number of GPUs

#SBATCH --time=12:00:00 # Set a runtime for the job in HH:MM:SS

#SBATCH --mem=32000 # Set the amount of memory for the job in MB.

conda activate mres_proj

srun python /gws/nopw/j04/ai4er/users/jl2182/MRes_Research_Project/src_experiments/landcover_classification/rf_landcover_classification.py