
#include <iostream>
#include <chrono>
using namespace std;

#include <stdlib.h>

#include <base_fft.h>

#ifdef SINGLE_PREC
typedef float myreal;
myreal EPS = 5e-6;
#else
typedef double myreal;
myreal EPS = 5e-13;
#endif

char *dealiasing_coeff_char = getenv("FLUIDFFT_DEALIASING_COEFF");

// template <typename T, typename U>
// inline std::complex<T> operator*(std::complex<T> lhs, const U& rhs)
// {
//     return lhs *= rhs;
// }

// template <typename T, typename U>
// inline std::complex<T> operator/(std::complex<T> lhs, const U& rhs)
// {
//     return lhs /= rhs;
// }

int are_nearly_equal(myreal a, myreal b) {
  if (abs((a - b) / a) > EPS)
    return 0;
  else
    return 1;
}

void BaseFFT::_init_parallel() {
  // cout << "BaseFFT::_init_parallel()";
  rank = 0;
  nb_proc = 1;
}

void BaseFFT::_init() {
  this->_init_parallel();

  if (this->are_parameters_bad())
    // if (rank == 0)
    throw invalid_argument("Invalid arguments");

  if (dealiasing_coeff_char != NULL)
    dealiasing_coeff = atof(dealiasing_coeff_char);
  if ((dealiasing_coeff == 0) || (dealiasing_coeff < 0) ||
      (dealiasing_coeff > 1))
    dealiasing_coeff = 1;

  if (rank == 0) {
    cout << endl << "--------" << endl;
    if (nb_proc > 1)
      cout << "nb_proc: " << nb_proc << endl;
  }
}

bool BaseFFT::are_parameters_bad() { return 0; }

char const *BaseFFT::get_classname() { return "BaseFFT"; }

myreal BaseFFT::compute_energy_from_X(myreal *fieldX) { return 0.; }

myreal BaseFFT::compute_energy_from_K(mycomplex *fieldK) { return 0.; }

myreal BaseFFT::compute_mean_from_X(myreal *fieldX) { return 0.; }

myreal BaseFFT::compute_mean_from_K(mycomplex *fieldK) { return 0.; }

int BaseFFT::test() {
  int OK = 1;
  myreal *fieldX;
  mycomplex *fieldK;
  myreal energy_X_before, energy_K_before, energy_K_after;
  myreal mean_X_before, mean_K_before, mean_K_after;

  if (rank == 0)
    cout << "tests (" << this->get_classname() << ")..." << endl;

  this->init_array_X_random(fieldX);
  this->alloc_array_K(fieldK);

  fft(fieldX, fieldK);

  // nicer initialization to kill some modes
  ifft(fieldK, fieldX);

  mean_X_before = this->compute_mean_from_X(fieldX);
  mean_K_before = this->compute_mean_from_K(fieldK);

  energy_X_before = this->compute_energy_from_X(fieldX);
  energy_K_before = this->compute_energy_from_K(fieldK);

  if (mean_X_before == 0. or mean_K_before == 0. or energy_X_before == 0. or
      energy_K_before == 0.) {
    cout << "Warning: tests not implemented for this class "
         << this->get_classname() << endl;
    OK = 0;
  }

  ifft(fieldK, fieldX);

  mean_K_after = this->compute_mean_from_K(fieldK);
  energy_K_after = this->compute_energy_from_K(fieldK);

  if (rank == 0) {
    if (!are_nearly_equal(energy_X_before, energy_K_before)) {
      printf("fail: (energy_X_before - energy_K_before)/energy_X_before = %e > "
             "EPS\n",
             abs((energy_X_before - energy_K_before) / energy_X_before));
      printf("      energy_X_before = %e\n", abs(energy_X_before));
      OK = 0;
    }
    if (!are_nearly_equal(mean_X_before, mean_K_before)) {
      printf("fail: (mean_X_before - mean_K_before)/mean_X_before = %e > EPS\n",
             abs((mean_X_before - mean_K_before) / mean_X_before));
      OK = 0;
    }

    if (!are_nearly_equal(energy_K_after, energy_K_before)) {
      cout << "fail: energy_K_after - energy_K_before > EPS" << endl;
      OK = 0;
    }

    if (!are_nearly_equal(mean_K_after, mean_K_before)) {
      cout << "fail: mean_K_after - mean_K_before > EPS" << endl;
      OK = 0;
    }

    if (OK)
      cout << " OK!" << endl;
  }

  free(fieldX);
  free(fieldK);
  return OK;
}

void BaseFFT::bench(int nb_time_execute, myreal *times) {
  int i;
  chrono::duration<double> duration_in_sec;
  myreal time_in_sec;
  myreal *fieldX;
  mycomplex *fieldK;
  char tmp_char[80];

  if (rank == 0)
    cout << "bench from cpp..." << endl;

  this->alloc_array_X(fieldX);
  this->alloc_array_K(fieldK);

  auto start_time = chrono::high_resolution_clock::now();
  for (i = 0; i < nb_time_execute; i++) {
    fieldX[0] = i;
    fft(fieldX, fieldK);
  }
  auto end_time = chrono::high_resolution_clock::now();

  if (rank == 0) {
    duration_in_sec = end_time - start_time;
    time_in_sec = duration_in_sec.count() / nb_time_execute;
    times[0] = time_in_sec;
    snprintf(tmp_char, sizeof(tmp_char), "time fft (%s):  %f s\n",
             this->get_classname(), time_in_sec);
    cout << tmp_char;
  }

  start_time = chrono::high_resolution_clock::now();
  for (i = 0; i < nb_time_execute; i++) {
    fieldK[0] = i;
    ifft(fieldK, fieldX);
  }
  end_time = chrono::high_resolution_clock::now();

  if (rank == 0) {
    duration_in_sec = end_time - start_time;
    time_in_sec = duration_in_sec.count() / nb_time_execute;
    snprintf(tmp_char, sizeof(tmp_char), "time ifft (%s): %f s\n",
             this->get_classname(), time_in_sec);
    cout << tmp_char;
    times[1] = time_in_sec;
  }

  free(fieldX);
  free(fieldK);
}

void BaseFFT::alloc_array_X(myreal *&fieldX) {
  fieldX = (myreal *)malloc(this->get_local_size_X() * sizeof(myreal));
}

void BaseFFT::alloc_array_K(mycomplex *&fieldK) {
  fieldK = (mycomplex *)malloc(this->get_local_size_K() * sizeof(mycomplex));
}

void BaseFFT::init_array_X_random(myreal *&fieldX) {
  int ii;
  // cout << "BaseFFT::init_array_X_random" << endl;
  this->alloc_array_X(fieldX);
  for (ii = 0; ii < this->get_local_size_X(); ++ii)
    fieldX[ii] = (myreal)rand() / RAND_MAX;
}

void BaseFFT::fft(myreal *fieldX, mycomplex *fieldK) {
  cout << "BaseFFT::fft" << endl;
}

void BaseFFT::ifft(mycomplex *fieldK, myreal *fieldX) {
  cout << "BaseFFT::ifft" << endl;
}

int BaseFFT::get_local_size_X() {
  cout << "BaseFFT::get_local_size_X" << endl;
  return nX0loc * nX1;
}

int BaseFFT::get_local_size_K() {
  cout << "BaseFFT::get_local_size_K" << endl;
  return nKxloc * nKy;
}
