#ifndef FFTW_FFT_HPP
#define FFTW_FFT_HPP

#include "types.hpp"
#include "fftw3.h"

extern "C"

{

/**
 * Perform in-place N1 batched complex one-dimensional FFTs, each of size N2 elements.
 * sign = -1 -> FFTW_FORWARD
 * sign = +1 -> FFTW_BACKWARD
 */
void batched1dffts(fcomp * data, int N1, int N2, int sign);

} // end extren "C"

#endif
