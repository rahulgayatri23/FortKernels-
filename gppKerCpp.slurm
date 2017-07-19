#!/bin/bash
#SBATCH -N 1
#SBATCH -p debug
#SBATCH -A m1759
#SBATCH -t 00:30:00 
#SBATCH -J rahuljob
#SBATCH -o rahuljob.o%j
#SBATCH -C haswell

echo "Running with 1 MPI ranks and 64 OMP_THREADS and 256 logical cores "

# OpenMP runtime parameters
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 1 -c 2 --cpu_bind=cores ./gppKerCpp 512 2 16384 20
#./gppKerCpp 512 2 8192 20
