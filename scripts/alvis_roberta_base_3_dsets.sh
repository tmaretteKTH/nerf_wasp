#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A NAISS2024-22-1455
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A40:3
#SBATCH --job-name=roberta_base_all_dsets
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-1455/project-data/nerf_wasp/logs/roberta_base_3_dsets_%A.out
#SBATCH -t 0-03:00:00


set -eo pipefail

module purge
module load Python/3.10.8-GCCcore-12.2.0
source /mimer/NOBACKUP/groups/naiss2024-22-1455/venvs/venv_example_lightning/bin/activate

# run the script from the nerf_wasp dir
# have 4 datasets
# TODO: switch FROM default = "local-simulation-gpu" TO default = "local-simulation-gpu-3"
# (in pyproject.toml)
flwr run . --run-config 'model-name="FacebookAI/xlm-roberta-base"'
