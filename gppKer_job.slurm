#!/bin/bash
#SBATCH -N 1
#SBATCH -p debug
#SBATCH -A m1759
#SBATCH -S 4
#SBATCH -t 00:20:00 

export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=24

srun -n 1 -c 48 --cpu_bind=cores ./gppKerCpp 512 2 2048 20 1
#for mpi_rank in 1 2 4 8 24; do
#    threads_per_rank=$((24/${mpi_rank}))
#    export OMP_NUM_THREADS=${threads_per_rank}
#    logical_cores=$(( ${threads_per_rank}*2 ))
#    echo "mpi_rank = ${mpi_rank} threads_per_rank = ${threads_per_rank} logical_cores = ${logical_cores}"
#    echo "export OMP_NUM_THREADS=${threads_per_rank}"
#    echo "srun -n ${mpi_rank} -c ${logical_cores} --cpu_bind=cores ./sw4lite_c skinny-rev.in"
#
#    srun -n ${mpi_rank} -c ${logical_cores} --cpu_bind=cores ./sw4lite_c skinny-rev.in
#done
