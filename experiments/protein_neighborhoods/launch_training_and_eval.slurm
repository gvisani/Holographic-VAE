#!/bin/bash
#SBATCH --job-name=hvae-z=128-lmax=6-v2
#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=110G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gvisan01@uw.edu
#SBATCH --export=all
#SBATCH -e slurm_logs/hvae-z=128-lmax=6-v2.err
#SBATCH -o slurm_logs/hvae-z=128-lmax=6-v2.out

python launch_training_and_eval.py --experiment_dir runs/hvae-z=128-lmax=6-v2 --config ../../config/hvae/hvae.yaml 
