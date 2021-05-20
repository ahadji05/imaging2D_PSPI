#ifndef PHASE_SHIFT_HPP
#define PHASE_SHIFT_HPP

#include "types.hpp"

extern "C"
{

#if defined(PWM_ENABLE_CUDA) || defined(PWM_ENABLE_HIP)

__global__ void phase_shift_forw_cu(int , int , int , float * , float * , float , fcomp * );

__global__ void phase_shift_back_cu(int , int , int , float * , float * , float , fcomp * );

__global__ void extrap_ref_wavefields_cu(int , int , int , int , fcomp * , fcomp * , int * , fcomp * );

#endif

void phase_shift_forw(int , int , int , fcomp * , float * , float * , float );

void phase_shift_back(int , int , int , fcomp * , float * , float * , float );

void extrap_ref_wavefields(fcomp * , fcomp * , fcomp * , int * , int, int , int , int );

}

#endif