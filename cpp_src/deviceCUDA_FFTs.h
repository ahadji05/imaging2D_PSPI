#ifndef deviceCUDA_FFTs_h
#define deviceCUDA_FFTs_h

#include <iostream>
#include <thrust/complex.h>
#include <cufft.h>
#include <cmath>

extern "C"
{

#define NRANK 1 // signals are 1-dimensional (NX spatial points).

void cufftFORW_Batched1dSignals(thrust::complex<float> * d_signals, int Nbatch, int NX);
void cufftBACK_Batched1dSignals(thrust::complex<float> * d_signals, int Nbatch, int NX);

} // end extern "C"

#endif

