#!/bin/bash
#SBATCH --job-name=hae-z=256=lmax=6-collected_lmax=6-rst_norm=square
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-23:00:00
#SBATCH --mem=110G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gvisan01@uw.edu
#SBATCH --export=all
#SBATCH -e slurm_logs/hae-z=256=lmax=6-collected_lmax=6-rst_norm=square.err
#SBATCH -o slurm_logs/hae-z=256=lmax=6-collected_lmax=6-rst_norm=square.out

python train_and_eval.py --model_dir runs/hae-z=256=lmax=6-collected_lmax=6-rst_norm=square --training_config config.yaml
