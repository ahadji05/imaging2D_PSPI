#ifndef IMAGING_CONDITIONS_HPP
#define IMAGING_CONDITIONS_HPP

#include "types.hpp"

extern "C"

{

#if defined(PWM_ENABLE_CUDA) || defined(PWM_ENABLE_HIP)

__global__ void imaging_cu(fcomp * ,fcomp * ,fcomp * ,int ,int ,int ,int ,int );

#endif

void imaging(int , int , int , int , fcomp * , fcomp * , int , int , float * );

} // end extern "C"

#endif