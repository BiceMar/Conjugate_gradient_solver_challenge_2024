#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=10                       # number of tasks
#SBATCH --qos=default                      # SLURM qos
#SBATCH --cpus-per-task=16                  # number of cores per task
#SBATCH --time=00:15:00                    # time (HH:MM:SS)
#SBATCH --partition=cpu                    # partition
#SBATCH --account=p200301                  # project account

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun  --mpi=pspmix --cpus-per-task=$SLURM_CPUS_PER_TASK ./conjugate_gradients
