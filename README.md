# Conjugate_gradient_solver_challenge_2024
EUMaster4HPC Student Challenge 2024.
Parallel implementation of the conjugate gradient method using a MPI and OpenMP.

Follow these steps to allocate resources, load the required modules, and submit the job on the system.

### 1. Allocate Resources

```sh
salloc -A p200301 --res cpudev -q dev -N 1 -t 00:30:00
```

### 2. Load Modules
```sh
module load intel
module load foss
```

### 3. Compile the program
```sh
mpic++ -O2 src/conjugate_gradients.cpp -o conjugate_gradients
```

### 4. Batch Script
Create a shell script (mpi_job.sh) for your SLURM job. Use the following template for the script:
```sh

#!/bin/bash -l
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=10                        # Number of tasks
#SBATCH --qos=default                      # SLURM Quality of Service
#SBATCH --cpus-per-task=16                 # Number of cores per task
#SBATCH --time=00:15:00                    # Time limit (HH:MM:SS)
#SBATCH --partition=cpu                    # Partition name
#SBATCH --account=p200301                  # Project account

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --mpi=pspmix --cpus-per-task=$SLURM_CPUS_PER_TASK ./conjugate_gradients

```
### 5. Submit your job

```sh
sbatch mpi_job.sh
```
