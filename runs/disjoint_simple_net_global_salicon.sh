#!/bin/bash
#SBATCH --job-name=dsnglbslc
#SBATCH --output=/scratch/izar/poletto/logs/log_disjoint_simple_net_global_salicon_%j.out
#SBATCH --error=/scratch/izar/poletto/logs/log_disjoint_simple_net_global_salicon_%j.err
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
srun python src/disjoint_simple_net/disjoint_simple_net_train.py -c /home/poletto/code/config/disjoint_simple_net/global_salicon.yml -n 4


