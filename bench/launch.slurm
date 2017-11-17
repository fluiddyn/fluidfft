#!/bin/bash
#SBATCH -J bench_fluidfft
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --ntasks-per-node=28
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00
#SBATCH --constraint=BDW28
#SBATCH --output bench_fluidfft_fftw.out

###SBATCH --qos=bonus

module purge

# to be able to compile cpp pythran code
module load gcc/6.2.0

module load intel/17.2
module load openmpi/intel/2.0.2
module load qt
module load hdf5-seq
module load python/3.6.3
unset PYTHONPATH

source ~/mypy/bin/activate

OPT=$HOME/opt

export LD_LIBRARY_PATH=$OPT/fftw/3.3.7/lib:$OPT/p3dfft/2.7.5/lib:$OPT/pfft/lib:$LD_LIBRARY_PATH

srun --mpi=pmi2 -K1 --resv-ports -n $SLURM_NTASKS fluidfft-bench 672 -d 3
