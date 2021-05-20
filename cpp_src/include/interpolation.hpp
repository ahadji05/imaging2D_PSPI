#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include <iostream>
#include <cmath>

#include "types.hpp"
#include "prep_operators.hpp"

extern "C"
{

#if defined(PWM_ENABLE_CUDA) || defined(PWM_ENABLE_HIP)

__global__ void interpolate_cu(int , int , int , int , float * , fcomp * , fcomp * );

#endif

void interpolate(int , int , int , int , float * , fcomp * , fcomp * );

} // end extern "C"

#endif