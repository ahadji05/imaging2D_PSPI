#ifndef DEVICE_ROCm_FFT_HPP
#define DEVICE_ROCm_FFT_HPP

#include "types.hpp"
#include "hipfft.h"

extern "C"
{

#define NRANK 1 // signals are 1-dimensional (NX spatial points).

hipfftHandle make_hipFFTplan_Batched1dSignals( int Nbatch, int NX );
void         hipfftFORW_Batched1dSignals( fcomp * d_signals, hipfftHandle * plan );
void         hipfftBACK_Batched1dSignals( fcomp * d_signals, int Nbatch, int NX, hipfftHandle * plan );

} // end extern "C"

#endif