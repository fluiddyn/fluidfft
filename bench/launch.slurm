#!/bin/bash
#SBATCH -J bench_fluidfft
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --ntasks-per-node=28
#SBATCH --threads-per-core=1
#SBATCH --time=00:55:00
#SBATCH --constraint=BDW28
#SBATCH --output bench_fluidfft.out

###SBATCH --qos=bonus

module purge
module load intel/17.2
module load openmpi/intel/2.0.2
module load qt
module load hdf5-seq
module load python/3.6.3
unset PYTHONPATH

# to be able to compile cpp pythran code
module load gcc/6.2.0

source ~/mypy/bin/activate

srun --mpi=pmi2 -K1 --resv-ports -n $SLURM_NTASKS fluidfft-bench 672 -d 3
