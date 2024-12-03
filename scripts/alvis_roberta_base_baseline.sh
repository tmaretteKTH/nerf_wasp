#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A NAISS2024-22-1455
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=alvis_roberta_base_baseline
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-1455/project-data/nerf_wasp/logs/roberta_base_all_dsets_%A.out
#SBATCH -t 0-08:00:00


set -eo pipefail

module purge
module load Python/3.10.8-GCCcore-12.2.0
source /mimer/NOBACKUP/groups/naiss2024-22-1455/venvs/venv_example_lightning/bin/activate

python src/run_baseline.py --train_datasets da_ddt --test_dataset da_ddt
python src/run_baseline.py --train_datasets sv_talbanken --test_dataset sv_talbanken
python src/run_baseline.py --train_datasets nno_norne --test_dataset nno_norne
python src/run_baseline.py --train_datasets nob_norne --test_dataset nob_norne

python src/run_baseline.py --train_datasets da_ddt sv_talbanken --test_dataset da_ddt
python src/run_baseline.py --train_datasets da_ddt sv_talbanken --test_dataset sv_talbanken

python src/run_baseline.py --train_datasets da_ddt sv_talbanken nno_norne --test_dataset da_ddt
python src/run_baseline.py --train_datasets da_ddt sv_talbanken nno_norne --test_dataset sv_talbanken
python src/run_baseline.py --train_datasets da_ddt sv_talbanken nno_norne --test_dataset nno_norne

python src/run_baseline.py --train_datasets da_ddt sv_talbanken nno_norne nob_norne --test_dataset da_ddt
python src/run_baseline.py --train_datasets da_ddt sv_talbanken nno_norne nob_norne --test_dataset sv_talbanken
python src/run_baseline.py --train_datasets da_ddt sv_talbanken nno_norne nob_norne --test_dataset nno_norne
python src/run_baseline.py --train_datasets da_ddt sv_talbanken nno_norne nob_norne --test_dataset nob_norne