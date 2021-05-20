#ifndef DEVICE_CUDA_FFT_HPP
#define DEVICE_CUDA_FFT_HPP

#include "types.hpp"
#include "cufft.h"

extern "C"
{

#define NRANK 1 // signals are 1-dimensional (NX spatial points).

cufftHandle  make_cuFFTplan_Batched1dSignals( int Nbatch, int NX );
void         cufftFORW_Batched1dSignals( fcomp * d_signals, cufftHandle * plan );
void         cufftBACK_Batched1dSignals( fcomp * d_signals, int Nbatch, int NX, cufftHandle * plan );

} // end extern "C"

#endif