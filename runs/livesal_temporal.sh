#!/bin/bash
#SBATCH --job-name=livesal
#SBATCH --output=/scratch/izar/poletto/logs/log%j.out
#SBATCH --error=/scratch/izar/poletto/logs/log%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Number of GPUs per node
#SBATCH --gres=gpu:1         # Adjust based on available GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=1:00:00

module purge
module load gcc
module load python

source /home/poletto/venvs/pdm/bin/activate

cd /home/poletto/code
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python src/livesal/livesal_train.py -c ../configs/livesal/temporal.yml


