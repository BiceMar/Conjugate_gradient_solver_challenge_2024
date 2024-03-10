# Conjugate_gradient_solver_challenge_2024
EUMaster4HPC Student Challenge 2024: parallel implementation of the conjugate gradient method using a MPI and OpenMP.

The main program in this project is `conjugate_gradients.cpp`, which solves the system. It loads and input dense matrix in row-major format and a right-hand-side from given binary files, performs the conjugate gradient iterations until convergence, and then writes the found solution to a given output binary file. A symmetric positive definite matrix and a right-hand-side can be generated using the `random_spd_system.sh` script and program.

Follow these steps to allocate resources, load the required modules, and submit the job on the system (MeluXina supercomputer).

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
#SBATCH --nodes=10                          # Number of nodes
#SBATCH --ntasks=10                        # Number of tasks
#SBATCH --qos=default                      # SLURM Quality of Service
#SBATCH --cpus-per-task=64                 # Number of cores per task
#SBATCH --time=00:15:00                    # Time limit (HH:MM:SS)
#SBATCH --partition=cpu                    # Partition name
#SBATCH --account=                         # Project account

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

srun --mpi=pspmix --cpus-per-task=$SLURM_CPUS_PER_TASK ./conjugate_gradients

```
### 5. Submit your job

```sh
sbatch mpi_job.sh
```
To generate a random SPD system of 10000 equations and unknowns:

```sh
./random_spd_system.sh 10000 io/matrix.bin io/rhs.bin

```
