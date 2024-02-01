
#include <iostream>
using namespace std;

#include <base_fftmpi.h>

void BaseFFTMPI::_init_parallel() {
  // cout << "BaseFFTMPI::_init_parallel()";
  /*DETERMINE RANK OF THIS PROCESSOR*/
  MPI_Comm_rank(MPI_COMM_WORLD, &(rank));
  /*DETERMINE TOTAL NUMBER OF PROCESSORS*/
  MPI_Comm_size(MPI_COMM_WORLD, &(nb_proc));
}
