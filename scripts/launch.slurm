#!/bin/bash
#SBATCH -J testfft
#SBATCH --nodes=2
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=24
#SBATCH --threads-per-core=1
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output test_fluidfft

module purge
module load libffi python/2.7.11 qt mkl
module load intel
module load bullxmpi/1.2.8.4
module load gcc/4.9.2
unset PYTHONPATH
source /home/bonamy/mypy/bin/activate
export FFTW3_LIB_DIR=/home/bonamy/fftw/lib/
export FFTW3_INC_DIR=/home/bonamy/fftw/include/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bonamy/fftw/lib/
export LANG=C
export LC_LANG=C
export OMP_NUM_THREADS=1
cd /home/bonamy/fluidfft/scripts
srun --mpi=pmi2 -K1 --resv-ports -n $SLURM_NTASKS python benchs3d.py 
