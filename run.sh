#!/bin/bash
#SBATCH --partition=gpu              # Partition (job queue)
#SBATCH --job-name=uhh           # Assign an short name to your job
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:1                 # Request number of GPUs
#SBATCH --constraint=pascal          # Specify hardware models
#SBATCH --mem=16000                  # Real memory (RAM) required (MB)
#SBATCH --time=00:05:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=./outputs/out_%u_%x_%j.out     # STDOUT output file
#SBATCH --error=./errors/out_%u_%x_%j.err

module purge

python3 main.py