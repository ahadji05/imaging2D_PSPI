#ifndef TYPES_HPP
#define TYPES_HPP

#if defined(PWM_ENABLE_CUDA) || defined(PWM_ENABLE_HIP)
    #include <thrust/complex.h>
#else
    #include "complex.h"
#endif

extern "C"
{

#if defined(PWM_ENABLE_CUDA) || defined(PWM_ENABLE_HIP)

    typedef thrust::complex<float> fcomp;
    namespace pwm = thrust;

#else

    typedef std::complex<float> fcomp;
    namespace pwm = std;

#endif

} // end extern "C"

#endif