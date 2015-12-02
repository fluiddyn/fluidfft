
#include <iostream>
using namespace std;

#include <stdlib.h>
#include <sys/time.h>

#include <base_fft.h>

double EPS = 1e-10;

double compute_time_in_second(struct timeval start_time,
			      struct timeval end_time)
{
  return (end_time.tv_sec - start_time.tv_sec) + 
         (end_time.tv_usec - start_time.tv_usec)/1000000.;
}


int are_nearly_equal(double a, double b)
{
  if (abs((a-b) / a) > EPS)
    return 0;
  else
    return 1;
}


void BaseFFT::_init_parallel()
{
  // cout << "BaseFFT::_init_parallel()";
  rank = 0;
  nb_proc = 1;
}


void BaseFFT::_init()
{
  this->_init_parallel();

  if (rank == 0)
    {
      cout << endl << "--------" << endl;
      if (nb_proc > 1)
	cout << "nb_proc: " << nb_proc << endl;
    }
}

char const* BaseFFT::get_classname()
{ return "BaseFFT";}

double BaseFFT::compute_energy_from_X(double* fieldX)
{
  return 0.;
}

double BaseFFT::compute_energy_from_K(fftw_complex* fieldK)
{
  return 0.;
}

double BaseFFT::compute_mean_from_X(double* fieldX)
{
  return 0.;
}

double BaseFFT::compute_mean_from_K(fftw_complex* fieldK)
{
  return 0.;
}

int BaseFFT::test()
{
  int OK = 1;
  double* fieldX;
  fftw_complex* fieldK;

  double energy_X_before, energy_K_before, energy_K_after;
  double mean_X_before, mean_K_before, mean_K_after;

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

  if (mean_X_before == 0. or
      mean_K_before == 0. or
      energy_X_before == 0. or
      energy_K_before == 0.)
    {
      cout << "Warning: tests not implemented for this class "
	   << this->get_classname() << endl;
      OK = 0;
    }

  ifft(fieldK, fieldX);
  
  mean_K_after = this->compute_mean_from_K(fieldK);
  energy_K_after = this->compute_energy_from_K(fieldK);
  
  if (!are_nearly_equal(energy_X_before, energy_K_before))
    {
      printf("fail: energy_X_before - energy_K_before = %e > EPS\n",
	     abs(energy_X_before - energy_K_before));
            printf("      energy_X_before = %e\n", abs(energy_X_before));
      OK = 0;
    }
  if (!are_nearly_equal(mean_X_before, mean_K_before))
    {
      printf("fail: mean_X_before - mean_K_before = %e > EPS\n",
	     abs(mean_X_before - mean_K_before));      
      OK = 0;
    }

  if (!are_nearly_equal(energy_K_after, energy_K_before))
    {
      cout << "fail: energy_K_after - energy_K_before > EPS" << endl;
      OK = 0;
    }
  
  if (!are_nearly_equal(mean_K_after, mean_K_before))
    {
      cout << "fail: mean_K_after - mean_K_before > EPS" << endl;
      OK = 0;
    }

  if (OK and rank == 0)
    cout << " OK!" << endl;
  
  free(fieldX);
  free(fieldK);
  return OK;
}


const char* BaseFFT::bench(int nb_time_execute)
{
  int i;
  struct timeval start_time, end_time;
  double time_in_sec;
  double* fieldX;
  fftw_complex* fieldK;
  string result("");
  char tmp_char[80];
  
  if (rank == 0) cout << "bench from cpp..." << endl;

  this->alloc_array_X(fieldX);
  this->alloc_array_K(fieldK);

  gettimeofday(&start_time, NULL);
  for (i=0; i<nb_time_execute; i++)
    {
        fft(fieldX, fieldK);
	// fieldX[0] = i;
    }
  gettimeofday(&end_time, NULL);

  if (rank == 0)
    {
      time_in_sec = compute_time_in_second(start_time, end_time) /
          nb_time_execute;
      snprintf(tmp_char, sizeof(tmp_char),
	       "time fft (%s):  %f s\n", this->get_classname(), time_in_sec);
      cout << tmp_char;
      result.append(tmp_char);
    }

  gettimeofday(&start_time, NULL);
  for (i=0; i<nb_time_execute; i++)
    {
        ifft(fieldK, fieldX);
	// fieldX[0] = i;
    }
  gettimeofday(&end_time, NULL);

  if (rank == 0)
    {
      time_in_sec = compute_time_in_second(start_time, end_time) /
          nb_time_execute;
      snprintf(tmp_char, sizeof(tmp_char),
	       "time ifft (%s): %f s\n", this->get_classname(), time_in_sec);
      cout << tmp_char;
      result.append(tmp_char);
    }
  
  free(fieldX);
  free(fieldK);
  return result.c_str();
}

void BaseFFT::alloc_array_X(double* &fieldX)
{  
  fieldX = (double *) malloc(this->get_local_size_X() * sizeof(double));
}


void BaseFFT::alloc_array_K(fftw_complex* &fieldK)
{
  fieldK = (fftw_complex*) malloc(
      this->get_local_size_K() * sizeof(fftw_complex));
}


void BaseFFT::init_array_X_random(double* &fieldX)
{
  cout << "BaseFFT::init_array_X_random" << endl;
  this->alloc_array_X(fieldX);
  fieldX[0] = 1;
}

void BaseFFT::fft(double *fieldX, fftw_complex *fieldK)
{
  cout << "BaseFFT::fft" << endl;
}

void BaseFFT::ifft(fftw_complex *fieldK, double *fieldX)
{
  cout << "BaseFFT::ifft" << endl;
}


int BaseFFT::get_local_size_X()
{
  cout << "BaseFFT::get_local_size_X" << endl;
  return nX0loc * nX1;
}


int BaseFFT::get_local_size_K()
{
  cout << "BaseFFT::get_local_size_K" << endl;
  return nKxloc * nKy;
}
