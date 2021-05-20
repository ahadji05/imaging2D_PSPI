#ifndef MKL_FFT_HPP
#define MKL_FFT_HPP

#include "types.hpp"
#include "mkl.h"

extern "C"

{

void c64fft1dforw(fcomp * , int );
void c64fft1dback(fcomp * , int );
void fft1dforwardFrom2Darray(fcomp * , int , int , int );
void fft1dbackwardFrom2Darray(fcomp * , int , int , int );

} // end extren "C"

#endif