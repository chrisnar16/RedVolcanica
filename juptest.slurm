#!/bin/bash
#SBATCH -J JupVolcano
#SBATCH -n 28
#SBATCH --gres=gpu:2
#SBATCH -p quinde-G9-2680
#SBATCH -w apollo-2
#SBATCH --qos=extended
#SBATCH -o salida_jupiter.out
jupyter notebook --port 8898

