#!/bin/bash
#SBATCH --job-name=livesal_salicon
#SBATCH --output=/scratch/izar/poletto/logs/log_livesal_graph_salicon_%j.out
#SBATCH --error=/scratch/izar/poletto/logs/log_livesal_graph_salicon_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

module purge
module load gcc
module load python

source /home/poletto/venvs/pdm/bin/activate

cd /home/poletto/code
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python src/livesal/livesal_train.py -c /home/poletto/code/config/livesal/graph_salicon.yml -n 4


